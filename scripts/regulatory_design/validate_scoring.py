"""运行 Enformer 上游样例，并核查全部搜索/检查模型的输入输出与物种轨道。"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import traceback
from .prepare import digest
from .run import atomic_json


def check_enformer(root):
    import torch
    from enformer_pytorch import Enformer
    from enformer_pytorch.modeling_enformer import Attention
    t0=time.perf_counter()
    model=Enformer.from_pretrained(str(root/'models/enformer-official-rough'),local_files_only=True).eval().cuda()
    fixture=root/'upstream_validation/enformer_test_sample.pt'
    data=torch.load(fixture,map_location='cpu',weights_only=False)
    results={}
    with torch.inference_mode():
        for tf_gamma in (False,):
            for module in model.modules():
                if isinstance(module,Attention):
                    module.use_tf_gamma=tf_gamma
            corr=model(data['sequence'].cuda(), target=data['target'].cuda(), return_corr_coef=True,head='human')
            results[str(tf_gamma)]=float(corr.cpu())
        # 上游样例长 131,072，只支持 use_tf_gamma=False；正式输入 196,608 才有 1,536 bins。
        from .protocol import Background
        from .scoring import one_hot
        background=Background(root)
        for module in model.modules():
            if isinstance(module,Attention):
                module.use_tf_gamma=True
        sequence=background.scorer_input(background.downstream[:3072],'enformer')
        production=model(torch.from_numpy(one_hot(sequence))[None].cuda(),head='mouse')
        production_ok=tuple(production.shape)==(1,896,1643) and bool(torch.isfinite(production).all())
    del model,data
    torch.cuda.empty_cache()
    return {'status':'passed' if all(v>.1 for v in results.values()) and production_ok else 'failed',
            'upstream_threshold':.1,'human_sample_pearson':results,
            'fixture_sha256':digest(fixture),'elapsed_s':time.perf_counter()-t0,
            'production_tf_gamma_full_input_passed':production_ok,
            'note':'原样复现上游 131,072 bp / tf_gamma=False 样例；另检查正式 196,608 bp / tf_gamma=True mouse 输出，不能对短样例强开 TF gamma。'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',default='data/phase5_assets')
    p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--only-enformer',action='store_true')
    p.add_argument('--output',default='benchmarks/phase5/scorer_validation.json')
    args=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import torch
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    root=Path(args.assets)
    out={'status':'running','gpu':args.gpu,'torch':torch.__version__,'checks':{}}
    atomic_json(args.output,out)
    try:
        out['checks']['enformer']=check_enformer(root)
        atomic_json(args.output,out)
        if not args.only_enformer:
            from .scoring import EnsembleScorer
            from .protocol import score_predictions, pattern, auc
            import numpy as np
            scorer=EnsembleScorer(root)
            sequence=scorer.background.downstream[:3072]
            e,f=scorer.predict(sequence)
            b=scorer.predict(sequence,check=True)
            # 尚未获得完整输入与参考数组，不能声称完成上游 TF 对拍。
            out['checks']['borzoi_upstream_fixture']={
                'status':'unavailable',
                'note':'公开 notebook 引用 wt_seq.npy 和 borzoi_wt_pred_across_folds.npy；尚未取得完整验证资产，未复现 TF 数值对拍。'}
            out['checks']['official_api_example']={
                'status':'passed' if e.shape==(896,) and f.shape==(4,6144) and b.shape==(6144,) and np.isfinite(e).all() and np.isfinite(f).all() and np.isfinite(b).all() else 'failed',
                'enformer_shape':list(e.shape),'flashzoi_shape':list(f.shape),'borzoi_shape':list(b.shape),
                'enformer_track':scorer.e_index,'borzoi_track':scorer.b_index,
                'natural_control_score':score_predictions(e,f,'medium',3072),
                'checker_control_auc':auc(pattern('medium',3072,32),b[:96])}
            out['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            out['checks']['coordinate_tests']='scripts/tests/test_regulatory_design.py'
            # 四条不同天然片段检查 batch 行映射，同时测定共用评分 batch。
            # 归一化全轨道的 L∞ 容差在应用校准前固定；不是生物质量等价界限。
            sequences=[scorer.background.downstream[i*4096:i*4096+3072] for i in range(4)]
            refs=[scorer.predict(s) for s in sequences]
            reference_e=np.stack([r[0] for r in refs])
            reference_f=np.stack([r[1] for r in refs])
            arrays={'scalar_enformer':reference_e,'scalar_flashzoi':reference_f}
            def normalized_error(actual,expected):
                dims=tuple(range(1,actual.ndim))
                a=actual/np.maximum(actual.max(axis=dims,keepdims=True),1e-12)
                b=expected/np.maximum(expected.max(axis=dims,keepdims=True),1e-12)
                return float(np.max(np.abs(a-b)))
            scorer.evaluate([sequence],'medium')
            batches=[]
            for batch in (1,2,4):
                try:
                    scorer.batch=batch
                    predictions=[scorer.predict_many(sequences[i:i+batch]) for i in range(0,4,batch)]
                    be=np.concatenate([p[0] for p in predictions])
                    bf=np.concatenate([p[1] for p in predictions])
                    arrays[f'b{batch}_enformer']=be
                    arrays[f'b{batch}_flashzoi']=bf
                    errors={'enformer':normalized_error(be,reference_e),'flashzoi':normalized_error(bf,reference_f)}
                    if not all(np.isfinite(v) and v<=.01 for v in errors.values()):
                        raise ValueError(f'评分 batch={batch} 超过预定 0.01 归一化轨道误差：{errors}')
                    times=[]
                    for repeat in range(2):
                        measured=scorer.evaluate(sequences,'medium')
                        times.append(measured['wall_s'])
                    batches.append({'batch':batch,'status':'complete','seconds_per_four':times,
                                    'normalized_profile_max_abs_error':errors,
                                    'peak_allocated_bytes':measured['peak_allocated_bytes']})
                except torch.OutOfMemoryError as exc:
                    batches.append({'batch':batch,'status':'oom','error':str(exc)})
                    torch.cuda.empty_cache()
            out['score_batch_calibration']=batches
            evidence=Path(args.output).with_suffix('.npz')
            np.savez_compressed(evidence,**arrays)
            out['checks']['batch_consistency']={'status':'passed','normalized_profile_linf_margin':.01,
                'evidence':str(evidence),'sha256':digest(evidence),'n_distinct_sequences':4}
            out['selected_score_batch']=min([b for b in batches if b['status']=='complete'],
                                            key=lambda b:np.median(b['seconds_per_four']))['batch']
            out['status']='passed' if out['checks']['enformer']['status']=='passed' and out['checks']['official_api_example']['status']=='passed' else 'failed'
            out['validation_scope']='Enformer 上游定量样例 + 六评分模型官方 API + 冻结坐标/轨道/shape/有限值检查；未完成 Borzoi 上游 TF 逐位复现，单独披露。'
        else:
            out['status']=out['checks']['enformer']['status']
        atomic_json(args.output,out)
    except Exception as exc:
        out.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        atomic_json(args.output,out)
        raise


if __name__=='__main__':
    main()
