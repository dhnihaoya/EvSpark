"""完整设计运行入口；加载/预热另计，所有候选和失败落盘。"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from pathlib import Path
import platform
import time
import traceback
import numpy as np

from .protocol import PROTOCOL, CALIBRATION_FOR, Background, freeze, canonical_hash
from .prepare import digest
from .scoring import ScorerClient

IDENTITY_FIELDS = ('backend','batch','pattern','seed','length','stage','protocol_sha256',
                   'assets_sha256','checkpoint_sha256','target_sha256',
                   'implementation_sha256','devices','score_batch')


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def candidate_seed(design_seed, round_index, parent_rank, candidate_index):
    return int(np.random.SeedSequence([design_seed, round_index, parent_rank, candidate_index]).generate_state(1, dtype=np.uint64)[0])


def implementation_hash():
    root=Path(__file__).resolve().parents[2]
    files=[root/'scripts/regulatory_design'/name for name in ('generation.py','scoring.py','protocol.py','run.py')]
    files += [root/p for p in ('evspark/train/drafter.py','evspark/specdec/block/driver.py','evspark/specdec/block/slice.py',
                              'evspark/specdec/block/chunk.py','evspark/specdec/block/neural_draft.py',
                              'evspark/specdec/verifier.py','evspark/specdec/transforms.py')]
    return canonical_hash({str(path.relative_to(root)):digest(path) for path in sorted(files)})


def run(args):
    import torch
    from evspark.specdec.block.ar_draft import load_evo2_7b_bf16
    from .generation import CandidateGenerator
    torch.set_num_threads(8)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'result.json'
    checkpoint_path = root / 'progress.json'
    if path.exists() and json.loads(path.read_text()).get('status') == 'complete':
        raise ValueError(f'{path} 已完成；禁止覆盖，换输出目录或直接汇总')
    protocol_sha = freeze(root / 'protocol.json')
    manifest_path = Path(args.assets) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for model in manifest['models'].values():
        if set(model['files']) != set(model['remote']):
            raise ValueError('模型资产尚未齐备')
    if len(manifest['models']) != 6:
        raise ValueError('须齐备 Enformer、4 Flashzoi 和 1 Borzoi 模型')
    # 校验资产独立于定时实验；结果冻结完整 manifest 与生成 checkpoint 哈希。
    for entry in list(manifest['files'].values()) + [f for m in manifest['models'].values() for f in m['files'].values()]:
        if digest(entry['path']) != entry['sha256']:
            raise ValueError(f"资产被修改：{entry['path']}")
    atomic_json(root / 'assets.json', manifest)
    record = {'status': 'running', 'backend': args.backend, 'batch': args.batch,
              'pattern': args.pattern, 'seed': args.seed, 'length': args.length, 'stage': args.stage,
              'protocol_sha256': protocol_sha, 'assets_sha256': digest(manifest_path),
              'checkpoint_sha256': digest(args.checkpoint),
              'target_sha256':digest(args.target_weights),
              'implementation_sha256':implementation_hash(),
              'devices': {'generation': os.environ.get('CUDA_VISIBLE_DEVICES'), 'scoring': args.scorer_gpu},
              'score_batch':args.score_batch,
              'resumed': bool(args.resume), 'timing_eligible': not args.resume,
              'environment': {'python':platform.python_version(), 'torch':torch.__version__,
                              'cuda':torch.version.cuda, 'gpu':torch.cuda.get_device_name(),
                              'packages':{d.metadata['Name']: d.version for d in importlib.metadata.distributions()}},
              'loading_s': {}, 'warmup_s': {}}
    prior=[json.loads(p.read_text()) for p in sorted((root/'attempts').glob('*.json'))]
    record['prior_attempts']=[{k:r.get(k) for k in ('status','error','failed_workflow_elapsed_s','elapsed_incomplete')} for r in prior]
    record['budget_eligible']=not any(r.get('elapsed_incomplete') for r in prior)
    record['prior_failed_elapsed_s']=sum(r.get('failed_workflow_elapsed_s',0.) for r in prior)
    atomic_json(path, record)
    scorer = gen = None
    try:
        t0 = time.perf_counter()
        scorer = ScorerClient(args.scorer_python, args.assets, args.scorer_gpu, root/'scorer.log',args.score_batch)
        record['loading_s']['scorer_process'] = time.perf_counter()-t0
        record['scorer_environment'] = scorer.ready
        t0 = time.perf_counter()
        model = load_evo2_7b_bf16(local_path=args.target_weights)
        gen = CandidateGenerator(model, args.checkpoint, batch=args.batch,
                                 max_length=40960+args.length+256,native_only=args.backend=='native')
        record['loading_s']['generator'] = time.perf_counter()-t0
        background = Background(args.assets)
        t0 = time.perf_counter()
        warm_state = gen.start(background.prompt[-1024:])
        gen.generate_candidates(warm_state, list(range(100,100+args.batch)), length=16, backend=args.backend)
        record['warmup_s']['generator'] = time.perf_counter()-t0
        del warm_state
        t0 = time.perf_counter()
        # 用天然背景截取的固定片段预热，非试验搜索候选；两臂完全相同。
        warm_sequence = background.downstream[:3072]
        scorer.request([warm_sequence], 'medium')
        scorer.request([warm_sequence], 'medium', check=True)
        record['warmup_s']['scorer'] = time.perf_counter()-t0
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        gen.meter.wall.clear()
        gen.meter.gpu.clear()
        stage_s = {'scoring':0., 'final_check':0., 'recovery':0.}
        scorer_cuda_s = {'scoring':0., 'final_check':0.}
        scorer_peak = 0
        started = time.perf_counter()
        record['workflow_started_unix']=time.time()
        atomic_json(path,record)
        if args.resume:
            saved = json.loads(checkpoint_path.read_text())
            expected = {k: record[k] for k in IDENTITY_FIELDS}
            if saved['identity'] != expected:
                raise ValueError('断点身份或资产不符，拒绝继续')
            t0 = time.perf_counter()
            beams = [(b['sequence'], gen.start(background.prompt+b['sequence']), b['id']) for b in saved['beams']]
            stage_s['recovery'] = time.perf_counter()-t0
            first_round = saved['round']+1
            final_guide = saved['final_guide']
            record['prior_partial_elapsed_s'] = saved['elapsed_s']
        else:
            beams = [('', gen.start(background.prompt), 'root')]
            first_round = 0
        trace_path = root/'trajectory.jsonl'
        if args.resume:
            # 丢弃已写轨迹但尚未提交 checkpoint 的尾部，避免重复轮次。
            with open(trace_path, 'r+b') as handle:
                handle.truncate(saved['trace_bytes'])
        mode = 'at' if args.resume else 'wt'
        with open(trace_path, mode, encoding='utf-8') as trace:
            for round_index in range(first_round, args.length//128):
                candidates = []
                for rank,(sequence,state,parent_id) in enumerate(beams):
                    seeds = [candidate_seed(args.seed, round_index, rank, i) for i in range(15)]
                    generated = gen.generate_candidates(state, seeds, length=128, backend=args.backend)
                    for index,candidate in enumerate(generated):
                        candidates.append({'sequence':sequence+candidate.sequence, 'suffix':candidate.sequence,
                                           'state':candidate.state, 'seed':candidate.seed, 'rng_state':candidate.rng_state,
                                           'sampling_trace':candidate.trace,
                                           'parent_id':parent_id, 'id':f'{round_index}:{rank}:{index}'})
                t0 = time.perf_counter()
                scores = scorer.request([c['sequence'] for c in candidates], args.pattern)
                stage_s['scoring'] += time.perf_counter()-t0
                scorer_cuda_s['scoring'] += scores['cuda_event_s']
                scorer_peak = max(scorer_peak, scores['peak_allocated_bytes'])
                valid=[i for i,s in enumerate(scores['scores']) if s['loss'] is not None]
                if len(valid)<2:
                    atomic_json(root/'failed_round.json', {'round':round_index,
                        'candidates':[{k:v for k,v in c.items() if k!='state'} | {'score':s}
                                      for c,s in zip(candidates,scores['scores'],strict=True)]})
                    raise RuntimeError('本轮不足两个有效 DNA 候选，不能保留预设两个分支')
                order = sorted(valid, key=lambda i:(scores['scores'][i]['loss'], i))
                keep = order[:2]
                beams = [(candidates[i]['sequence'], candidates[i]['state'], candidates[i]['id']) for i in keep]
                final_guide = scores['scores'][keep[0]]
                trace_record = {'round':round_index, 'generated_bp':128*(round_index+1),
                                'retained_ids':[candidates[i]['id'] for i in keep],
                                'candidates':[{k:v for k,v in c.items() if k not in ('state','sequence')}
                                              | {'score':s} for c,s in zip(candidates,scores['scores'], strict=True)]}
                trace.write(json.dumps(trace_record)+'\n')
                trace.flush()
                os.fsync(trace.fileno())
                elapsed = time.perf_counter()-started
                atomic_json(checkpoint_path, {'identity':{k:record[k] for k in IDENTITY_FIELDS},
                                             'round':round_index,'elapsed_s':elapsed, 'final_guide':final_guide,
                                             'trace_bytes':trace.tell(),
                                             'beams':[{'sequence':s,'id':identity} for s,_,identity in beams]})
                print(json.dumps({'round':round_index+1,'elapsed_s':elapsed,'loss':final_guide['loss']}), flush=True)
                # 不保留淘汰候选的私有状态；选中分支只引用共同祖先。
                del candidates, generated, scores
        t0 = time.perf_counter()
        checked = scorer.request([beams[0][0]], args.pattern, check=True)
        stage_s['final_check'] = time.perf_counter()-t0
        scorer_cuda_s['final_check'] = checked['cuda_event_s']
        scorer_peak = max(scorer_peak, checked['peak_allocated_bytes'])
        torch.cuda.synchronize()
        total = time.perf_counter()-started
        stage_s.update(gen.meter.wall)
        # recovery 重建前缀包含于 prefill/cache，单独标识而不重复相加。
        stage_s['orchestration_and_io'] = total-sum(v for k,v in stage_s.items() if k!='recovery')
        checker = checked['scores'][0]
        guide_auc, check_auc = final_guide['ensemble_auc'], checker['checker_auc']
        record.update(status='complete', total_s=total, stages_s=stage_s,
                      budget_total_s=total+record['prior_failed_elapsed_s'],
                      generator_cuda_event_s=gen.meter.gpu, scorer_cuda_event_s=scorer_cuda_s,
                      allocated_gpu_seconds=2*total,
                      peak_generation_bytes=torch.cuda.max_memory_allocated(), peak_scorer_bytes=scorer_peak,
                      sequence=beams[0][0], guide=final_guide, checker=checker,
                      qualified=(guide_auc is not None and check_auc is not None and guide_auc>=.9 and check_auc>=.9))
        (root/'final.fasta').write_text(f'>{args.backend}_{args.pattern}_{args.seed}\n'+beams[0][0]+'\n')
        atomic_json(path, record)
        return record
    except Exception as exc:
        record.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        if 'started' in locals():
            record['failed_workflow_elapsed_s']=time.perf_counter()-started
        atomic_json(path, record)
        raise
    finally:
        if gen is not None:
            gen.close()
        if scorer is not None:
            scorer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['native','evspark'], required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--pattern', choices=['medium','short','long','arc'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--length', type=int, required=True)
    p.add_argument('--stage', choices=[*CALIBRATION_FOR.values(),*CALIBRATION_FOR], required=True)
    p.add_argument('--assets', default='data/phase5_assets')
    p.add_argument('--checkpoint', default='benchmarks/phasec_ckpts/L27_g12_150M_s1.pt')
    p.add_argument('--target-weights',default='models/evo2_7b/evo2_7b.pt')
    p.add_argument('--scorer-python', default='/tmp/evspark_phase5_env/bin/python')
    p.add_argument('--scorer-gpu', type=int, default=1)
    p.add_argument('--score-batch',type=int,default=1)
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    allowed = PROTOCOL[args.stage]
    patterns = allowed.get('patterns', [allowed.get('pattern')])
    seeds = allowed.get('seeds', [allowed.get('seed')])
    if args.pattern not in patterns or args.seed not in seeds or args.length != allowed['length']:
        p.error('运行参数不符合冻结阶段协议')
    if args.batch not in PROTOCOL[args.backend+'_batches']:
        p.error('batch 不在预设扫描范围')
    if args.stage in ('formal','long_validation','calibration_formal','calibration_long'):
        gate = json.loads(Path('benchmarks/phase5/pilot_gate.json').read_text())
        if gate.get('passed') is not True:
            p.error('pilot 未过门，禁止扩大')
    run(args)


if __name__ == '__main__':
    main()
