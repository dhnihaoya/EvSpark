"""实验配置、坐标与评分的纯 CPU 实现；参数在首次运行前冻结。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

PROTOCOL = {
    'version': 2, 'model': 'evo2_7b', 'checkpoint': 'L27_g12_150M_s1',
    'gamma': 12, 'temperature': 1.0, 'top_k': 4, 'prompt_bp': 40960,
    'chunk_bp': 128, 'candidates_per_parent': 15, 'beam_width': 2,
    'assembly': 'mm39', 'chrom': 'chrX', 'replacement_0based': [52051928, 52123468],
    'track_accession': 'ENCFF872IES', 'replicate_std_ddof': 0,
    'pilot': {'patterns': ['medium', 'short'], 'seeds': [2026091901, 2026091902], 'length': 3072},
    'calibration': {'pattern': 'medium', 'seed': 2026091801, 'length': 3072},
    'calibration_formal': {'pattern': 'medium', 'seed': 2026091802, 'length': 6144},
    'calibration_long': {'pattern': 'long', 'seed': 2026091803, 'length': 19968},
    'formal': {'patterns': ['medium', 'short', 'long'], 'seeds': list(range(2026092001, 2026092009)), 'length': 6144},
    'long_validation': {'patterns': ['long', 'arc'], 'seeds': [2026092101, 2026092102], 'length': 19968},
    'pilot_gate': {'checker_mean_auc_each_arm': .80, 'median_paired_speedup': 1.25},
    'qualification': {'ensemble_auc': .90, 'checker_auc': .90},
    'native_batches': [1, 2, 4, 8], 'evspark_batches': [1, 2, 4],
}

CALIBRATION_FOR = {'pilot':'calibration','formal':'calibration_formal','long_validation':'calibration_long'}


def selection_filename(calibration):
    return 'selected_batches.json' if calibration=='calibration' else f'selected_batches_{calibration}.json'


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def freeze(path):
    path = Path(path)
    payload = {'protocol': PROTOCOL, 'sha256': canonical_hash(PROTOCOL)}
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError('已有协议不同；禁止覆盖冻结协议')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    return payload['sha256']


def pattern(name, length, bin_bp):
    if length <= 0 or length % bin_bp:
        raise ValueError('长度必须是正的完整 bin 数')
    if name in ('medium', 'short', 'long'):
        on, off = {'medium': (768, 768), 'short': (384, 1152), 'long': (1664, 1664)}[name]
        bases = (np.arange(length) % (on + off) < on).astype(np.float64)
    elif name == 'arc':
        units = []
        for letter in ['.-', '.-.', '-.-.']:
            if units:
                units.extend([0] * 3)
            for i, mark in enumerate(letter):
                if i:
                    units.append(0)
                units.extend([1] * (1 if mark == '.' else 3))
        bases = np.repeat(units, 384)
        bases = np.pad(bases, (0, max(0, length - len(bases))))[:length]
    else:
        raise ValueError(f'未知图案：{name}')
    bins = bases.reshape(-1, bin_bp)
    if not np.all(bins == bins[:, :1]):
        raise ValueError('图案边缘跨 bin，不能作为二元 AUROC 标签')
    return bins[:, 0]


def auc(labels, scores):
    """秩和 AUROC；并列使用平均秩，单一类别返回 None（早期搜索常见）。"""
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.shape != s.shape or not np.isfinite(s).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('AUROC 输入无效')
    n1 = int(y.sum())
    n0 = len(y) - n1
    if not n1 or not n0:
        return None
    order = np.argsort(s, kind='stable')
    values = s[order]
    starts = np.r_[0, np.flatnonzero(np.diff(values)) + 1]
    ends = np.r_[starts[1:], len(s)]
    ranks = np.empty(len(s), dtype=float)
    for lo, hi in zip(starts, ends):
        ranks[order[lo:hi]] = (lo + 1 + hi) / 2
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def normalized(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('评分模型输出须为非负有限数')
    maximum = float(values.max())
    if maximum <= 0:
        raise ValueError('评分轨道全为零')
    return values / maximum


def score_predictions(enformer, flashzoi, name, length):
    e, f = np.asarray(enformer), np.asarray(flashzoi)
    if e.shape != (896,) or f.shape != (4, 6144):
        raise ValueError(f'输出形状错误：{e.shape}, {f.shape}')
    if length % 128 or length > 896 * 128:
        raise ValueError('设计长度不适配输出区域')
    ep = normalized(e)[:length // 128]
    fn = normalized(f)
    fp = (fn.mean(axis=0) - fn.std(axis=0, ddof=PROTOCOL['replicate_std_ddof']))[:length // 32]
    ey, fy = pattern(name, length, 128), pattern(name, length, 32)
    ea, fa = auc(ey, ep), auc(fy, fp)
    return {'loss': float(.5 * (np.abs(ey - ep).sum() + np.abs(fy - fp).sum())),
            'enformer_auc': ea, 'flashzoi_auc': fa,
            'ensemble_auc': None if ea is None or fa is None else .5 * (ea + fa)}


class Background:
    def __init__(self, root):
        root = Path(root)
        self.upstream = json.loads((root / 'upstream.json').read_text())['dna'].upper()
        self.downstream = json.loads((root / 'downstream.json').read_text())['dna'].upper()
        if len(self.upstream) != 163840 or len(self.downstream) < 360448:
            raise ValueError('参考片段长度不符')

    @property
    def prompt(self):
        return self.upstream[-40960:]

    def scorer_input(self, sequence, model):
        left, size = (40960, 196608) if model == 'enformer' else (163840, 524288)
        right = size - left - len(sequence)
        if right < 0 or right > len(self.downstream):
            raise ValueError('设计长度不适配评分输入')
        result = self.upstream[-left:] + sequence + self.downstream[:right]
        assert len(result) == size
        return result


def track_index(path, accession='ENCFF872IES'):
    # Enformer 文件第一列沿用 human+mouse 全局编号；模型 mouse head 使用行序号。
    lines = Path(path).read_text().strip().splitlines()[1:]
    found = [i for i, line in enumerate(lines) if accession in line.split('\t')]
    if len(found) != 1:
        raise ValueError(f'{path}: 轨道 {accession} 匹配 {len(found)} 行')
    return found[0]
