"""正式 40k 前缀下的 batch 可行性与纯生成剖析；不能用于选择最终 batch。"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import torch
from .generation import CandidateGenerator
from .protocol import Background
from .run import atomic_json
from evspark.specdec.block.ar_draft import load_evo2_7b_bf16


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='benchmarks/phase5/generation_batch_probe.json')
    parser.add_argument('--batches',type=int,nargs='+',default=[1,2,4,8])
    parser.add_argument('--design-length',type=int,default=3072)
    args=parser.parse_args()
    torch.set_num_threads(8)
    model=load_evo2_7b_bf16()
    prompt=Background('data/phase5_assets').prompt
    result={'kind':'generation_only_feasibility_not_full_workflow_calibration','rows':[]}
    for batch in args.batches:
        for backend in ('native','evspark'):
            if backend=='evspark' and batch==8:
                continue
            gen=None
            row={'backend':backend,'batch':batch,'max_length':40960+args.design_length+256}
            try:
                gen=CandidateGenerator(model,'benchmarks/phasec_ckpts/L27_g12_150M_s1.pt',batch=batch,
                                       max_length=row['max_length'],native_only=backend=='native')
                root=gen.start(prompt)
                gen.generate_candidates(root,list(range(501,501+batch)),length=16,backend=backend)
                gen.meter.wall.clear()
                gen.meter.gpu.clear()
                torch.cuda.reset_peak_memory_stats()
                t0=time.perf_counter()
                candidates=gen.generate_candidates(root,list(range(800,815)),length=128,backend=backend)
                torch.cuda.synchronize()
                elapsed=time.perf_counter()-t0
                counts=[step['consumed'] for c in candidates for step in c.trace]
                row.update(status='complete',wall_s=elapsed,stages_s=gen.meter.wall,
                           throughput=15*128/elapsed,mean_consumed=sum(counts)/len(counts),
                           peak_allocated_bytes=torch.cuda.max_memory_allocated())
                del candidates,root
            except torch.OutOfMemoryError as exc:
                row.update(status='oom',error=str(exc))
            except Exception:
                row.update(status='failed',error=traceback.format_exc())
                raise
            finally:
                if gen is not None:
                    gen.close()
                del gen
                gc.collect()
                torch.cuda.empty_cache()
                result['rows'].append(row)
                atomic_json(args.output,result)
                print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
