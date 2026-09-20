"""多轮双分支状态压力验证；无评分器，不作为应用效果或速度数据。"""
from __future__ import annotations
import argparse
import gc
import hashlib
import os
from pathlib import Path
import time
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ.setdefault('HF_HOME',str(Path(__file__).resolve().parents[2]/'hf_home'))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import torch
from evspark.specdec.block.ar_draft import load_evo2_7b_bf16
from .generation import CandidateGenerator
from .protocol import Background
from .run import atomic_json,candidate_seed


def fingerprint(state,*,only_root=False):
    h=hashlib.sha256()
    nodes=state.chain()
    if only_root:
        tensors=list(nodes[0].kv.values())
    else:
        tensors=[t for node in nodes[1:] for t in node.kv.values()]
        tensors += [t for attrs in state.hyena.values() for values in attrs.values() for t in values.values()]
        tensors += [state.context]
    for t in tensors:
        h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['native','evspark'],required=True)
    p.add_argument('--batch',type=int,required=True)
    p.add_argument('--rounds',type=int,default=24)
    p.add_argument('--candidates',type=int,default=8)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    torch.set_num_threads(8)
    out={'status':'running','kind':'branch_state_stress_only_not_application',**vars(args),'round_records':[]}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(path,out)
    gen=None
    try:
        model=load_evo2_7b_bf16()
        gen=CandidateGenerator(model,'benchmarks/phasec_ckpts/L27_g12_150M_s1.pt',batch=args.batch,
                               max_length=40960+max(6144,args.rounds*128)+256,native_only=args.backend=='native')
        root=gen.start(Background('data/phase5_assets').prompt)
        root_hash=fingerprint(root,only_root=True)
        beams=[([],root)]
        started=time.perf_counter()
        for step in range(args.rounds):
            children=[]
            for rank,(tokens,state) in enumerate(beams):
                before=fingerprint(state)
                seeds=[candidate_seed(2026091907,step,rank,i) for i in range(args.candidates)]
                generated=gen.generate_candidates(state,seeds,length=128,backend=args.backend)
                assert fingerprint(state)==before, '父分支状态发生写入'
                for c in generated:
                    assert len(c.token_ids)==128
                    assert c.state.length==40959+(step+1)*128
                    assert c.state.anchor==c.token_ids[-1]
                    children.append((tokens+c.token_ids,c.state))
            # 固定保留首尾：从第二轮起分别来自两个父分支，验证两条祖先链。
            beams=[children[0],children[-1]]
            del children,generated
            gc.collect()
            row={'round':step+1,'generated_bp':(step+1)*128,'parent_states_unchanged':True,
                 'chain_lengths':[len(s.chain()) for _,s in beams],
                 'allocated_bytes':torch.cuda.memory_allocated(),'elapsed_s':time.perf_counter()-started}
            out['round_records'].append(row)
            atomic_json(path,out)
            print(row,flush=True)
        assert fingerprint(root,only_root=True)==root_hash, '根前缀 KV 被改写'
        out.update(status='passed',root_kv_unchanged=True,final_tokens=[t for t,_ in beams],
                   peak_allocated_bytes=torch.cuda.max_memory_allocated())
        atomic_json(path,out)
    except Exception as exc:
        import traceback
        out.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        atomic_json(path,out)
        raise
    finally:
        if gen is not None:gen.close()


if __name__=='__main__':
    main()
