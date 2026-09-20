"""真实 7B 的独立候选、缓存隔离和续写检查（不进入效果统计）。"""
from __future__ import annotations
import argparse
import os
import sys
import json
from pathlib import Path
import time
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'regulatory_design'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parents[2] / 'hf_home'))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
from .generation import CandidateGenerator
from .protocol import Background
from evspark.specdec.block.ar_draft import load_evo2_7b_bf16
from evspark.specdec.block.driver import step_seqlen_offsets
from evspark.specdec.block.driver import block_forward
from evspark.specdec.block.slice import slice_states_to_accept
from evspark.specdec.transforms import apply_transform


@torch.inference_mode()
def forced_logits(gen, state, tokens):
    gen._restore(state,1)
    rows=[]
    for token in [state.anchor]+tokens[:-1]:
        logits,_=gen.model(torch.tensor([[token]],device=gen.device),inference_params_dict=gen.ip)
        rows.append(logits[0,-1].float().cpu().numpy())
        step_seqlen_offsets(gen.ip,1)
        gen.ip['mha'].lengths_per_sample.add_(1)
    return np.stack(rows)


def numerical_evidence(actual, reference):
    records=[]
    for pos,(a,b) in enumerate(zip(actual,reference)):
        ia,ib=np.argsort(-a,kind='stable'),np.argsort(-b,kind='stable')
        if ia[0]!=ib[0]:
            ag,bg=float(a[ia[0]]-a[ia[1]]),float(b[ib[0]]-b[ib[1]])
            records.append({'position':pos,'actual_top4':ia[:4].tolist(),'reference_top4':ib[:4].tolist(),
                            'actual_logits_top4':a[ia[:4]].tolist(),'reference_logits_top4':b[ib[:4]].tolist(),
                            'actual_gap':ag,'reference_gap':bg,'strict_near_tie_1e_3':min(ag,bg)<1e-3,
                            'max_abs_delta_logits':float(np.abs(a-b).max())})
    tv=[float(.5*np.abs(apply_transform(a)-apply_transform(b)).sum()) for a,b in zip(actual,reference)]
    return {'forced_prefix_flips':records,'max_tvd_top4':max(tv),'mean_tvd_top4':float(np.mean(tv)),
            'max_abs_delta_logits':float(np.abs(actual-reference).max())}


def classify_validation(out):
    """接线检查与 native/block 数值差异分开；near-tie 沿用项目原定 1e-3。"""
    core=[c for c in out['cases'] if c['kind'] in ('batched_greedy','resume_greedy')]
    exact_core=all(not any(c['mismatch_positions']) if c['kind']=='batched_greedy'
                   else not c['mismatch_positions'] for c in core)
    batch_checks=[c['scalar_same_schedule'] for c in out['cases'] if c.get('scalar_same_schedule') is not None]
    non_ties=sum(not flip['strict_near_tie_1e_3'] for case in batch_checks for flip in case['forced_prefix_flips'])
    out['recovery_non_near_tie_flips']=non_ties
    out['recovery_near_tie_flips']=sum(len(c['forced_prefix_flips']) for c in batch_checks)-non_ties
    out['correctness_passed']=bool(exact_core and out['native_sample_resume_equal'] and out['state_invariants']
                                    and out.get('native_optimized_equal',False)
                                    and len(batch_checks)==2 and non_ties==0)
    out['status']=('passed' if out['all_exact_greedy'] else 'passed_with_bf16_differences') if out['correctness_passed'] else 'needs_numerical_review'
    out['numerical_scope']='native 与 block 路径的全部差异保留；新增批量状态恢复以相同验证块的单流重放对照判定，不把路径数值差异归为缓存错误。'
    return out


@torch.inference_mode()
def replay_blocks(gen, initial, candidate):
    gen._restore(initial,1)
    context=initial.context.clone()
    length=initial.length
    for step in candidate.trace:
        gen.capture.begin()
        _,stash=block_forward(gen.model,torch.tensor([step['inputs']],device=gen.device),gen.ip,retain=True)
        h=gen._captured()[0,:step['consumed']].float()
        slice_states_to_accept(gen.ip,stash,step['consumed']-1)
        context=torch.cat([context,h])[-gen.drafter.ctx_window:]
        length+=step['consumed']
    return gen._extract(0,length,candidate.token_ids[-1],context,initial)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', default='data/phase5_assets')
    p.add_argument('--checkpoint', default='benchmarks/phasec_ckpts/L27_g12_150M_s1.pt')
    p.add_argument('--context', type=int, default=1024)
    p.add_argument('--output', default='benchmarks/phase5/generation_validation.json')
    args = p.parse_args()
    torch.set_num_threads(8)
    out = {'context': args.context, 'cases': [], 'status': 'running', 'torch': torch.__version__}
    path = Path(args.output)
    def save():
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False)+'\n')
    save()
    try:
        model = load_evo2_7b_bf16()
        gen = CandidateGenerator(model, args.checkpoint, batch=4, max_length=args.context+4096)
        prompt = Background(args.assets).prompt[-args.context:]
        initial = gen.start(prompt)
        frozen = {k: v.clone() for k, v in initial.kv.items()}
        baseline = gen.generate_candidates(initial, [101], length=64, backend='native', greedy=True)[0]
        for backend in ('native', 'evspark'):
            rows = gen.generate_candidates(initial, [201,202,203,204], length=64, backend=backend, greedy=True)
            mismatches = [[j for j,(a,b) in enumerate(zip(row.token_ids, baseline.token_ids)) if a != b] for row in rows]
            isolation = all(torch.equal(initial.kv[k], v) for k,v in frozen.items())
            out['cases'].append({'backend': backend, 'kind': 'batched_greedy', 'batch': 4,
                                 'mismatch_positions': mismatches, 'parent_unchanged': isolation,
                                 'lengths': [row.state.length for row in rows],
                                 'tokens': [row.token_ids for row in rows], 'baseline': baseline.token_ids})
            save()
            first = gen.generate_candidates(initial, [301], length=32, backend=backend, greedy=True)[0]
            second = gen.generate_candidates(first.state, [302], length=32, backend=backend, greedy=True)[0]
            mismatch = [j for j,(a,b) in enumerate(zip(first.token_ids+second.token_ids, baseline.token_ids)) if a != b]
            out['cases'].append({'backend': backend, 'kind':'resume_greedy', 'mismatch_positions':mismatch,
                                 'length':second.state.length})
            save()
        # native 的随机数流可逐位精确续写；投机截断边界消耗 RNG 不同，不能要求同种子逐位相同。
        whole = gen.generate_candidates(initial, [501], length=64, backend='native')[0]
        first = gen.generate_candidates(initial, [501], length=32, backend='native')[0]
        second = gen.generate_candidates(first.state, [501], length=32, backend='native', rng_states=[first.rng_state])[0]
        out['native_sample_resume_equal'] = whole.token_ids == first.token_ids + second.token_ids
        for backend in ('native', 'evspark'):
            sampled = gen.generate_candidates(initial, [601,602,603,604], length=64, backend=backend)
            out['cases'].append({'backend':backend,'kind':'sampled_batch',
                                 'lengths':[len(c.token_ids) for c in sampled],
                                 'state_lengths':[c.state.length for c in sampled],
                                 'unique_sequences':len({tuple(c.token_ids) for c in sampled})})
            # 验证实际随机 batch 分叉后的状态，而不只检查各行相同的贪心路径。
            for index in (0,3):
                candidate = sampled[index]
                actual = gen.generate_candidates(candidate.state, [701], length=16, backend='native', greedy=True)[0]
                with torch.inference_mode():
                    gen._restore(initial,1)
                    context=initial.context.clone()
                    replay_ids=[initial.anchor]+candidate.token_ids[:-1]
                    for token in replay_ids:
                        gen.capture.begin()
                        model(torch.tensor([[token]],device=gen.device),inference_params_dict=gen.ip)
                        h=gen._captured()[0].float()
                        context=torch.cat([context,h])[-gen.drafter.ctx_window:]
                        step_seqlen_offsets(gen.ip,1)
                        gen.ip['mha'].lengths_per_sample.add_(1)
                    replay=gen._extract(0, initial.length+len(replay_ids),candidate.token_ids[-1],context,initial)
                reference=gen.generate_candidates(replay,[702],length=16,backend='native',greedy=True)[0]
                diagnostic=numerical_evidence(forced_logits(gen,candidate.state,reference.token_ids),
                                              forced_logits(gen,replay,reference.token_ids))
                block_check=None
                if backend=='evspark':
                    same_schedule=replay_blocks(gen,initial,candidate)
                    block_check=numerical_evidence(forced_logits(gen,candidate.state,reference.token_ids),
                                                   forced_logits(gen,same_schedule,reference.token_ids))
                out['cases'].append({'backend':backend,'kind':'sampled_state_continuation','row':index,
                                     'mismatch_positions':[j for j,(a,b) in enumerate(zip(actual.token_ids,reference.token_ids)) if a!=b],
                                     'actual':actual.token_ids,'native_replay':reference.token_ids,
                                     'numerical_evidence':diagnostic,'scalar_same_schedule':block_check})
                save()
        out['all_exact_greedy'] = all(not any(c['mismatch_positions']) if c['kind']=='batched_greedy'
                                     else not c['mismatch_positions'] for c in out['cases'] if 'mismatch_positions' in c)
        out['state_invariants'] = all(c.get('parent_unchanged',True) for c in out['cases']) and all(
            all(length==args.context+63 for length in c['state_lengths']) for c in out['cases'] if 'state_lengths' in c)
        out['timing'] = gen.meter.wall
        gen.close()
        native=CandidateGenerator(model,args.checkpoint,batch=4,max_length=args.context+4096,native_only=True)
        native_initial=native.start(prompt)
        native_greedy=native.generate_candidates(native_initial,[101],length=64,backend='native',greedy=True)[0]
        native_sample=native.generate_candidates(native_initial,[501],length=64,backend='native')[0]
        native_first=native.generate_candidates(native_initial,[501],length=32,backend='native')[0]
        native_second=native.generate_candidates(native_first.state,[501],length=32,backend='native',rng_states=[native_first.rng_state])[0]
        out['native_optimized_equal']=(native_greedy.token_ids==baseline.token_ids and native_sample.token_ids==whole.token_ids
                                      and native_first.token_ids+native_second.token_ids==whole.token_ids)
        out['native_optimized_scope']='不加载 drafter、不捕获/更新其 hidden；原生贪心、采样、分段 RNG 续写与通用实现对拍。'
        native.close()
        classify_validation(out)
        out['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        save()
        print(json.dumps({'status':out['status'],'timing':out['timing']}), flush=True)
    except Exception as exc:
        import traceback
        out['status'] = 'failed'
        out['error'] = repr(exc)
        out['traceback'] = traceback.format_exc()
        save()
        raise


if __name__ == '__main__':
    main()
