"""独立评分进程；CPU JSON 接口避免将评分依赖装进 evo2 环境。"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

from .protocol import Background, score_predictions, pattern, auc, track_index


class ScorerClient:
    def __init__(self, python, assets, gpu, log, score_batch=1):
        import os
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM='false',
                   OMP_NUM_THREADS='8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        self.log = open(log, 'a')
        self.process = subprocess.Popen([str(python), '-u', '-m', 'regulatory_design.scoring',
                                         '--assets', str(assets),'--batch',str(score_batch)], env=env, text=True,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
        self.ready = self._read()
        if self.ready.get('status') != 'ready':
            raise RuntimeError(self.ready)

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'评分进程退出，详见 {self.log.name}')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def request(self, sequences, pattern_name, *, check=False):
        self.process.stdin.write(json.dumps({'sequences': sequences, 'pattern': pattern_name, 'check': check}) + '\n')
        self.process.stdin.flush()
        return self._read()

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait(timeout=30)
        self.log.close()


def one_hot(sequence):
    table = np.zeros((256, 4), dtype=np.float32)
    for i, base in enumerate(b'ACGT'):
        table[base, i] = 1
    if set(sequence) - set('ACGTN'):
        raise ValueError('生成了非 ACGTN 字符；记录失败，不改变 target 分布过滤后重采样')
    return table[np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)]


class EnsembleScorer:
    def __init__(self, assets, batch=1):
        import torch
        from enformer_pytorch import Enformer
        from enformer_pytorch.modeling_enformer import Attention
        from borzoi_pytorch import Borzoi
        self.torch = torch
        self.batch=int(batch)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        self.root = Path(assets)
        self.background = Background(assets)
        self.e_index = track_index(self.root / 'enformer_targets.tsv')
        self.b_index = track_index(self.root / 'borzoi_targets.tsv')
        self.loading_info = {}
        def load(cls,name):
            model,info=cls.from_pretrained(str(self.root/'models'/name),local_files_only=True,output_loading_info=True)
            missing=[key for key in info.get('missing_keys',[]) if not key.endswith('num_batches_tracked')]
            if missing or info.get('unexpected_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
                raise ValueError(f'{name} 权重不完整或模型版本不符：{info}')
            self.loading_info[name]=info
            return model.eval().cuda()
        self.enformer = load(Enformer,'enformer-official-rough')
        # 上游便利函数仅在 repo 名精确匹配时启用 TF gamma；本地路径须显式设置。
        for module in self.enformer.modules():
            if isinstance(module, Attention):
                module.use_tf_gamma = True
        self.flash = [load(Borzoi,f'flashzoi-replicate-{i}') for i in range(4)]
        self.checker = load(Borzoi,'borzoi-replicate-0-mouse')
        if not all(m.enable_mouse_head and m.flashed for m in self.flash):
            raise ValueError('Flashzoi 权重不是启用 mouse head 的 Flashzoi')
        if not self.checker.enable_mouse_head or self.checker.flashed:
            raise ValueError('检查模型须为原版 Borzoi mouse')
        for model in [self.enformer, *self.flash, self.checker]:
            model.requires_grad_(False)

    def predict(self, sequence, *, check=False):
        result=self.predict_many([sequence],check=check)
        return result[0] if check else (result[0][0],result[1][0])

    def predict_many(self, sequences, *, check=False):
        torch = self.torch
        x=torch.from_numpy(np.stack([one_hot(self.background.scorer_input(s,'borzoi')).T for s in sequences])).cuda()
        with torch.inference_mode():
            if check:
                # 原版 Borzoi 按原实现 fp32 推理，不使用 Flashzoi 代替检查模型。
                pred = self.checker(x, is_human=False)[:, self.b_index].float().cpu().numpy()
                if pred.shape != (len(sequences),6144) or not np.isfinite(pred).all():
                    raise ValueError('检查模型输出无效')
                return pred
            flash = []
            for model in self.flash:
                with torch.autocast('cuda', dtype=torch.float16):
                    pred = model(x, is_human=False)[:, self.b_index]
                flash.append(pred.float().cpu().numpy())
            del x
            x=torch.from_numpy(np.stack([one_hot(self.background.scorer_input(s,'enformer')) for s in sequences])).cuda()
            # Enformer 使用 fp32 和 TF gamma，避免改动公开评分定义。
            e = self.enformer(x, head='mouse')[:, :, self.e_index].float().cpu().numpy()
        return e, np.stack(flash,axis=1)

    def evaluate(self, sequences, name, *, check=False):
        torch = self.torch
        torch.cuda.reset_peak_memory_stats()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        begin.record()
        output = [None]*len(sequences)
        valid=[]
        for index,sequence in enumerate(sequences):
            invalid=sorted(set(sequence)-set('ACGTN'))
            if invalid:
                if check:
                    raise ValueError(f'最终序列包含非 DNA 字符：{invalid}')
                output[index]={'loss':None,'ensemble_auc':None,'enformer_auc':None,'flashzoi_auc':None,
                               'invalid_reason':f'非 ACGTN 字符：{invalid}'}
                continue
            valid.append(index)
        for start in range(0,len(valid),self.batch):
            indices=valid[start:start+self.batch]
            predicted=self.predict_many([sequences[i] for i in indices],check=check)
            for row,index in enumerate(indices):
                sequence=sequences[index]
                if check:
                    pred=predicted[row]
                    output[index]={'checker_auc':auc(pattern(name,len(sequence),32),pred[:len(sequence)//32]),
                                   'checker_profile':pred[:len(sequence)//32].tolist()}
                else:
                    e,f=predicted[0][row],predicted[1][row]
                    score=score_predictions(e,f,name,len(sequence))
                    score.update(enformer_profile=e[:len(sequence)//128].tolist(),
                                 flashzoi_profiles=f[:,:len(sequence)//32].tolist(),
                                 enformer_normalizer=float(e.max()),flashzoi_normalizer=float(f.max()))
                    output[index]=score
        end.record()
        end.synchronize()
        return {'scores': output, 'wall_s': time.perf_counter()-t0,
                'cuda_event_s': begin.elapsed_time(end)/1000,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--batch',type=int,default=1)
    args = parser.parse_args()
    try:
        t0 = time.perf_counter()
        with redirect_stdout(sys.stderr):
            scorer = EnsembleScorer(args.assets,batch=args.batch)
        print(json.dumps({'status': 'ready', 'load_s': time.perf_counter()-t0,
                          'gpu': scorer.torch.cuda.get_device_name(), 'torch': scorer.torch.__version__,
                          'score_batch':scorer.batch,
                          'loading_info':scorer.loading_info,
                          'cudnn_allow_tf32':scorer.torch.backends.cudnn.allow_tf32,
                          'matmul_allow_tf32':scorer.torch.backends.cuda.matmul.allow_tf32,
                          'enformer_track': scorer.e_index, 'borzoi_track': scorer.b_index}), flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                result = scorer.evaluate(request['sequences'], request['pattern'], check=request.get('check', False))
            print(json.dumps(result), flush=True)
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({'error': repr(exc)}), flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
