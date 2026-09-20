"""实验完成后的入稿与数字核验；运行前强制核对投入门、全部设计及原始文件。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from .protocol import PROTOCOL,CALIBRATION_FOR,canonical_hash,selection_filename
from .report import load_results,paired,summarize
from .prepare import digest
from .run import atomic_json

OLD_SCOPE=r'''Our performance claims concern decode latency for a single sequence. Batch
microbenchmarks characterize forward-pass capacity, but independent-request
speculation still requires scheduling variable accepted lengths and managing
per-sequence state. The multi-candidate implementation is functional but
slower than one candidate in the tested setting. Prefill is excluded from
speedup measurements, so full-request gains will be smaller when prefill
dominates, particularly for short continuations after long prompts.'''
NEW_SCOPE=r'''The token-generation benchmarks concern single-sequence decode latency
and exclude prefill. The regulatory-design experiment additionally measures
complete designs, including prefill, batched independent candidates and scoring,
under a fixed beam-search policy. This application scheduler does not establish
performance for arbitrary mixtures of concurrent requests. The earlier
multi-candidate verifier produces one continuation and is distinct from these
independent design candidates. Full-request gains depend on the costs of
prefill, generation and the downstream scoring workload.'''
OLD_REMAINING='''Coding sequences have lower acceptance across target sizes. Within the tested
designs, increasing training positions or candidate count does not reliably
improve latency. Better coding-region proposals and parallel candidate sampling are
concrete directions for improvement. Their value should be judged by
end-to-end time, with training and data-collection costs included, rather
than by acceptance alone.'''
NEW_REMAINING='''Coding sequences have lower acceptance across target sizes. In the tested
single-output verifiers, increasing training positions or the number of draft
proposals does not reliably improve latency. Better coding-region proposals
and more efficient scheduling across independent requests remain directions
for improvement. Their value should be judged by end-to-end time, including
training and data-collection costs, rather than acceptance alone.'''

BIBLIOGRAPHY={
'enformer':r'''\bibitem{enformer}
\v{Z}. Avsec et al.
Effective gene expression prediction from sequence by integrating long-range interactions.
\emph{Nature Methods}, 18:1196--1203, 2021.
\url{https://doi.org/10.1038/s41592-021-01252-x}.
''',
'borzoi':r'''\bibitem{borzoi}
J. Linder et al.
Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation.
\emph{Nature Genetics}, 57:949--961, 2025.
\url{https://doi.org/10.1038/s41588-024-02053-6}.
''',
'flashzoi':r'''\bibitem{flashzoi}
J. C. Hingerl, A. Karollus, and J. Gagneur.
Flashzoi: an enhanced Borzoi for accelerated genomic analysis.
\emph{Bioinformatics}, 41(9):btaf467, 2025.
\url{https://doi.org/10.1093/bioinformatics/btaf467}.
'''}


def validated_stages(root):
    root=Path(root)
    gate=json.loads((root/'pilot_gate.json').read_text())
    if gate.get('passed') is not True or not gate.get('checks') or not all(gate['checks'].values()):
        raise ValueError('pilot 未过预设投入门，禁止入稿')
    selection=json.loads((root/selection_filename('calibration')).read_text())
    pilot=summarize(paired(load_results(root,'pilot'),'pilot',selection))
    if canonical_hash(pilot)!=canonical_hash(gate['summary']):
        raise ValueError('pilot 投入门与原始配对数据不符')
    thresholds=PROTOCOL['pilot_gate']
    if pilot['speedup_median']['estimate']<thresholds['median_paired_speedup'] or any(
            pilot['arms'][b]['mean_checker_auc']<thresholds['checker_mean_auc_each_arm'] for b in ('native','evspark')):
        raise ValueError('重算 pilot 未达到原定阈值')
    stages={}
    for stage in ('formal','long_validation'):
        selection=json.loads((root/selection_filename(CALIBRATION_FOR[stage])).read_text())
        pairs=paired(load_results(root,stage),stage,selection)
        actual=summarize(pairs)
        saved=json.loads((root/f'{stage}_summary.json').read_text())
        if canonical_hash(actual)!=canonical_hash(saved):
            raise ValueError(f'{stage} 汇总与原始数据不一致')
        stages[stage]={'pairs':pairs,'summary':actual,'selection':selection}
    return stages


def render(stages):
    f=stages['formal'];l=stages['long_validation'];s=f['summary'];ls=l['summary']
    speed=s['speedup_geometric_mean'];quality=s['checker_auc_delta']
    batches={b:f['selection']['selection'][b]['batch'] for b in ('native','evspark')}
    curve=s['arms'];counts={b:curve[b]['budget_curve']['qualified_unique'][-1] for b in curve}
    hours=s['common_budget_seconds']/3600
    def ci(value,places=2):
        return rf"$[{value['ci95'][0]:.{places}f},\,{value['ci95'][1]:.{places}f}]$"
    text=rf'''% PHASE5_RESULT_BEGIN
\subsection{{Regulatory DNA design under a fixed search policy}}
\label{{sec:regulatory-workflow}}
We evaluated a computational mouse chromatin-accessibility design task based
on Evo2's published workflow~\cite{{evo2}}. Both backends used Evo2 7B with
40,960\,bp of genomic context, generated 128\,bp per step, sampled 15 independent
candidates per retained branch, and kept the two best candidates globally.
Search combined Enformer~\cite{{enformer}} and four Flashzoi
replicates~\cite{{flashzoi}}; the original Borzoi mouse replicate-0
model~\cite{{borzoi}} was reserved for final checking. This checker was excluded
from search, without a claim of independent training data.

The main experiment comprised {len(f['pairs'])} paired designs of 6,144\,bp across
three predefined patterns and eight seeds. Separate full-workflow calibration selected native
batch {batches['native']} and EvSpark batch {batches['evspark']} from the prespecified
feasible batch grids on the same devices. Including prefill, cache management,
search scoring and final checking, the paired geometric-mean speedup was
{speed['estimate']:.2f}$\times$ (95\% CI {ci(speed)}).
The paired change in checker AUROC (EvSpark minus native) was
{quality['estimate']:+.3f} (95\% CI {ci(quality,3)}).

A qualified design required both the search-ensemble and final-checker AUROCs
to reach 0.90. Within the common observed budget of {hours:.2f}\,h,
EvSpark and native yielded {counts['evspark']} and {counts['native']} exact-unique
qualified final designs, respectively, in the predefined task order
(Figure~\ref{{fig:regulatory-workflow}}). Qualification denotes a predictive
criterion, not experimentally validated function. The {len(l['pairs'])} paired
19,968\,bp designs had a geometric-mean workflow speedup of
{ls['speedup_geometric_mean']['estimate']:.2f}$\times$
(95\% CI {ci(ls['speedup_geometric_mean'])}).

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/fig_regulatory_workflow.pdf}}
\caption{{Complete regulatory-DNA design workflows. (A) Mean elapsed time per
design, including prefill, cache handling, generation and scoring; model
loading and warmup are recorded separately. (B) Cumulative exact-unique
qualified final designs, restricted to the time interval covered by both
arms. (C) Final Borzoi checker AUROC, with lines connecting paired designs;
the dashed line marks the qualification threshold. (D) Descriptive aligned
sequence identity and pooled 4-mer entropy. Each design contributes only its
top-ranked final sequence. The sampling unit for confidence intervals is a
complete design, with all patterns sharing a seed resampled together; sequence
positions and pairwise identities are not treated as independent experimental replicates.}}
\label{{fig:regulatory-workflow}}
\end{{figure}}
\FloatBarrier
% PHASE5_RESULT_END
'''
    appendix=[r'% PHASE5_METHOD_BEGIN',r'\section{Regulatory-design protocol and resource use}',r'\label{app:regulatory}',
        r'Generation used the released L27/$\gamma=12$/150M/s1 drafter, temperature 1 and top-$k=4$. The mm39 chrX replacement interval was interpreted as the zero-based half-open interval [52,051,928, 52,123,468); predictor inputs combined the fixed upstream flank, generated sequence and downstream flank of that interval.',
        r'The target track was ENCFF872IES (ES-E14 DNase; mouse-head indices 11 in Enformer and 741 in Borzoi). Enformer used 196,608 bp inputs and 128 bp bins; Flashzoi/Borzoi used 524,288 bp inputs and 32 bp bins. Only generated bins entered the loss. Each Enformer profile was normalized by its full-profile maximum; Flashzoi used a joint maximum over four replicates, followed by their mean minus population standard deviation. The two masked L1 sums had equal weight. Search-ensemble AUROC was the mean of the Enformer and Flashzoi AUROCs.',
        r'The main patterns were 768/768, 384/1152 and 1664/1664 bp open/closed waves. Long designs used the 1664/1664 wave and ARC Morse pattern. ARC used 384 bp dots, 1152 bp dashes, 384 bp gaps within letters and 1152 bp gaps between letters, starting at the first generated base; the remaining 8832 bp were assigned closed chromatin. This placement and padding convention was fixed in the implementation. All model revisions, weights, reference sequences, track tables, actual seeds and candidate traces were frozen and retained. Non-ACGTN candidates were recorded as invalid without resampling; a design failed if fewer than two valid branches remained.',
        r'Native used batch 1/2/4/8 and EvSpark batch 1/2/4 calibration grids. Each design length had a separate calibration seed and full-workflow batch choice; the scorer batch and devices were shared. Each beam-search iteration completed candidate generation before scoring and global selection. Native neither loaded a drafter nor maintained drafter hidden states. Shared immutable prefixes and private incremental candidate states were used by both arms. The pilot used separate seeds, and its correctness, per-arm mean checker AUROC $\geq0.80$, and median paired speedup $\geq1.25$ gates were frozen before application runs.',
        r'Enformer and the original Borzoi checker used fp32; Flashzoi used fp16 autocast, with TF32 disabled in the scoring process. Enformer used the upstream numerical validation example and explicit TF gamma for full-length design inputs. The original Borzoi notebook comparison against TensorFlow reference predictions was not reproduced because the full reference fixture was unavailable; weight, species-head, coordinate, shape and batch-consistency checks were recorded separately.',
        r'Confidence intervals used 10,000 paired bootstrap resamples grouped by the reused design seed, retaining the complete designs for every pattern together (eight seed groups in the main experiment and two in long validation). This preserves cross-pattern dependence from shared early random trajectories; point estimates still weight each complete design equally. Long-validation intervals are descriptive given only two seed groups. An interval containing zero is not an equivalence result. Fixed-budget counts followed the prespecified run order and included recorded failed-attempt time. Interrupted runs with incomplete timing were ineligible for that analysis. Exact duplicates, aligned sequence identity and 4-mer entropy were computed from generated regions only.',
        r'\begin{table}[t]\centering\small',r'\caption{Observed resource use of complete designs. Memory is peak allocated GPU memory; elapsed time excludes model loading and warmup, which are reported separately.}',
        r'\begin{tabular}{llrrrr}\toprule',r'Design & Backend & Batch & Time (min) & Gen. GiB & Score GiB \\ \midrule']
    for name,stage in stages.items():
        for i,backend in enumerate(('native','evspark')):
            rows=[p[i] for p in stage['pairs']]
            mins=np.mean([r['total_s'] for r in rows])/60
            gm=max(r['peak_generation_bytes'] for r in rows)/2**30
            sm=max(r['peak_scorer_bytes'] for r in rows)/2**30
            length=rows[0]['length'];batch=rows[0]['batch']
            appendix.append(f'{length:,} bp & {backend} & {batch} & {mins:.2f} & {gm:.2f} & {sm:.2f} '+r'\\')
    appendix += [r'\bottomrule\end{tabular}\end{table}',r'% PHASE5_METHOD_END','']
    return text,'\n'.join(appendix)


def splice(text,fragment,begin,end,anchor):
    if begin in text or end in text:
        if text.count(begin)!=1 or text.count(end)!=1:raise ValueError('重复或残缺的 Phase 5 入稿标记')
        lo=text.index(begin);hi=text.index(end)+len(end)
        if hi-len(end)<lo:raise ValueError('Phase 5 入稿标记顺序错误')
        return text[:lo]+fragment.rstrip()+text[hi:]
    if text.count(anchor)!=1:raise ValueError(f'论文插入锚点不唯一：{anchor}')
    return text.replace(anchor,fragment+'\n'+anchor,1)


def materialize(root,project):
    root,project=Path(root),Path(project)
    stages=validated_stages(root)
    body,appendix=render(stages)
    (root/'paper_results.tex').write_text(body)
    (root/'paper_methods.tex').write_text(appendix)
    sources={r['source']:digest(r['source']) for stage in stages.values() for pair in stage['pairs'] for r in pair}
    index={'status':'rendered_from_complete_results','sources':sources,
           'summaries':{str(root/f'{stage}_summary.json'):digest(root/f'{stage}_summary.json') for stage in stages},
           'results_sha256':digest(root/'paper_results.tex'),'methods_sha256':digest(root/'paper_methods.tex')}
    main=project/'paper/latex_v3/main.tex'
    if main.exists():
        original=main.read_text()
        if OLD_SCOPE in original:updated=original.replace(OLD_SCOPE,NEW_SCOPE,1)
        elif NEW_SCOPE in original:updated=original
        else:raise ValueError('Serving scope 段已有其他修改，保留原文并停止自动入稿')
        if OLD_REMAINING in updated:updated=updated.replace(OLD_REMAINING,NEW_REMAINING,1)
        elif NEW_REMAINING not in updated:raise ValueError('Remaining performance 段已有其他修改，停止自动入稿')
        updated=splice(updated,body,'% PHASE5_RESULT_BEGIN','% PHASE5_RESULT_END',r'\section{Related work}')
        updated=splice(updated,appendix,'% PHASE5_METHOD_BEGIN','% PHASE5_METHOD_END',r'\begin{thebibliography}{10}')
        for key,entry in BIBLIOGRAPHY.items():
            if rf'\bibitem{{{key}}}' not in updated:
                updated=updated.replace(r'\end{thebibliography}',entry+'\n'+r'\end{thebibliography}',1)
        if main.read_text()!=original:raise ValueError('入稿时论文被另一进程修改，未覆盖')
        figures=main.parent/'figures';figures.mkdir(exist_ok=True)
        shutil.copyfile(root/'formal_main_figure.pdf',figures/'fig_regulatory_workflow.pdf')
        backup=root/'paper_before_phase5.tex'
        if not backup.exists():backup.write_text(original)
        tmp=main.with_suffix('.phase5.tmp');tmp.write_text(updated);tmp.replace(main)
        index.update(paper=str(main),paper_sha256=digest(main),paper_figure_sha256=digest(figures/'fig_regulatory_workflow.pdf'))
    atomic_json(root/'paper_materials.json',index)
    return index


def audit(root,project):
    root,project=Path(root),Path(project)
    index=json.loads((root/'paper_materials.json').read_text())
    for path,sha in index['sources'].items():
        if digest(path)!=sha:raise ValueError(f'原始结果已改变：{path}')
    for path,sha in index['summaries'].items():
        if digest(path)!=sha:raise ValueError(f'结果汇总已改变：{path}')
    body,appendix=render(validated_stages(root))
    if (root/'paper_results.tex').read_text()!=body or (root/'paper_methods.tex').read_text()!=appendix:
        raise ValueError('入稿数字/协议与原始结果重算不一致')
    if index.get('paper'):
        text=(project/'paper/latex_v3/main.tex').read_text()
        if body.rstrip() not in text or appendix.rstrip() not in text or NEW_SCOPE not in text or NEW_REMAINING not in text:
            raise ValueError('论文正文与核验片段不一致')
        figure=project/'paper/latex_v3/figures/fig_regulatory_workflow.pdf'
        if digest(figure)!=index['paper_figure_sha256']:raise ValueError('入稿主图被修改')
    return {'status':'passed','n_source_designs':len(index['sources'])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='benchmarks/phase5')
    p.add_argument('--project',default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--compile',action='store_true')
    args=p.parse_args()
    if not args.audit_only:materialize(args.root,args.project)
    atomic_json(Path(args.root)/'paper_audit.json',audit(args.root,args.project))
    main=Path(args.project)/'paper/latex_v3/main.tex'
    if args.compile and main.exists():
        binary=shutil.which('tectonic') or str(Path(sys.prefix).parent/'tex/bin/tectonic')
        if not Path(binary).exists():raise RuntimeError('未找到 tectonic；入稿文件保留，尚未完成编译验证')
        with open(Path(args.root)/'paper_compile.log','w') as log:
            subprocess.run([binary,'--keep-logs','main.tex'],cwd=main.parent,stdout=log,stderr=subprocess.STDOUT,check=True)
        atomic_json(Path(args.root)/'paper_compile.json',{'status':'passed','pdf_sha256':digest(main.with_suffix('.pdf'))})
    print('Phase 5 入稿数字重算通过。')


if __name__=='__main__':main()
