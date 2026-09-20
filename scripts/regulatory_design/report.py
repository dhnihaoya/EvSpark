"""配对统计、冻结投入门、固定预算曲线与主图；绝不补造缺失运行。"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from .protocol import PROTOCOL, CALIBRATION_FOR, selection_filename
from .run import atomic_json


def load_results(root, stage):
    records = []
    for path in Path(root).glob('*/result.json'):
        record = json.loads(path.read_text())
        if record.get('stage') == stage:
            record['source'] = str(path)
            records.append(record)
    return records


def select_batches(records):
    selection = {}
    for field in ('protocol_sha256','assets_sha256','checkpoint_sha256','target_sha256',
                  'implementation_sha256','score_batch'):
        if len({r[field] for r in records}) != 1:
            raise ValueError(f'校准 {field} 不一致，不能混用配置')
    for backend in ('native','evspark'):
        rows = [r for r in records if r['backend']==backend]
        expected = set(PROTOCOL[backend+'_batches'])
        measured = {r['batch'] for r in rows if r['status'] in ('complete','failed')}
        if measured != expected or len(rows)!=len(expected):
            raise ValueError(f'{backend} 批量校准不完整：{measured} != {expected}')
        valid = [r for r in rows if r['status']=='complete' and r['timing_eligible']]
        if any(not np.isfinite(r['total_s']) or r['total_s']<=0 for r in valid):
            raise ValueError('校准完整耗时必须为正且有限')
        if any(r['status']=='failed' and 'out of memory' not in r.get('error','').lower() for r in rows):
            raise ValueError('存在非 OOM 失败，须排除实现错误后重新校准')
        if not valid:
            raise ValueError(f'{backend} 无可用完整校准')
        best = min(valid, key=lambda r:r['total_s'])
        selection[backend] = {'batch':best['batch'], 'total_s':best['total_s'], 'source':best['source']}
    devices = {json.dumps(r['devices'], sort_keys=True) for r in records if r['status']=='complete'}
    if len(devices)!=1:
        raise ValueError('校准设备组合不同，不能比较')
    return {'selection':selection, 'devices':json.loads(next(iter(devices))),
            'attempts':[{k:r.get(k) for k in ('backend','batch','status','total_s','error','source')} for r in records]}


def paired(records, stage, selection):
    cfg = PROTOCOL[stage]
    keyed = {}
    for row in records:
        if row['status']!='complete' or not row['timing_eligible']:
            raise ValueError('存在失败/未完成/恢复运行，不能静默排除后形成完整结果')
        if row['length']!=cfg['length'] or len(row['sequence'])!=cfg['length']:
            raise ValueError('实际输出长度与阶段协议不符')
        if not np.isfinite(row['total_s']) or row['total_s']<=0:
            raise ValueError('完整耗时必须为正且有限')
        if row['batch'] != selection['selection'][row['backend']]['batch'] or row['devices'] != selection['devices']:
            raise ValueError('实际 batch/设备与完整工作流校准选择不符')
        key = (row['pattern'],row['seed'],row['backend'])
        if key in keyed:
            raise ValueError(f'设计重复记录：{key}')
        keyed[key] = row
    keys = [(p,s) for p in cfg['patterns'] for s in cfg['seeds']]
    expected = {(p,s,b) for p,s in keys for b in ('native','evspark')}
    if set(keyed)!=expected:
        raise ValueError(f'阶段记录不完整：缺少 {sorted(expected-set(keyed))}；多出 {sorted(set(keyed)-expected)}')
    for field in ('protocol_sha256','assets_sha256','checkpoint_sha256','target_sha256','implementation_sha256','score_batch'):
        if len({r[field] for r in records}) != 1:
            raise ValueError(f'两臂 {field} 不一致')
    return [(keyed[p,s,'native'],keyed[p,s,'evspark']) for p,s in keys]


def bootstrap(values, statistic=np.mean, n=10000, clusters=None):
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError('空统计样本')
    rng = np.random.default_rng(2026091905)
    if clusters is None:
        groups=np.arange(len(values))[:,None]
    else:
        labels=np.asarray(clusters)
        if labels.shape!=values.shape:raise ValueError('种子分组与设计数不符')
        groups=[np.flatnonzero(labels==label) for label in np.unique(labels)]
        if len({len(group) for group in groups})!=1:
            raise ValueError('每个种子须覆盖同样数量的图案，不能混入不完整分组')
        groups=np.stack(groups)
    # 同一设计种子在各图案下复用，早期前缀可能相同；整组抽取才能保留相关性。
    indices=groups[rng.integers(len(groups),size=(n,len(groups)))].reshape(n,-1)
    samples = values[indices]
    estimates = statistic(samples, axis=1)
    return {'estimate':float(statistic(values)), 'ci95':np.quantile(estimates,[.025,.975]).tolist(),
            'n_designs':len(values), 'n_seed_clusters':len(groups),
            'bootstrap_replicates':n,'bootstrap_seed':2026091905,'interval_method':'percentile',
            'unit':'完整设计；复用种子内各图案成组配对重采样'}


def diversity(sequences):
    if not sequences:
        return {'exact_unique':0,'aligned_identity':[],'fourmer_entropy_bits':None}
    counts = np.zeros(256)
    mapping = dict(zip('ACGT',range(4)))
    for sequence in sequences:
        for pos in range(len(sequence)-3):
            word=sequence[pos:pos+4]
            if all(base in mapping for base in word):
                index=0
                for base in word:
                    index=4*index+mapping[base]
                counts[index]+=1
    probs=counts[counts>0]/max(1,counts.sum())
    identities=[float(np.mean(np.frombuffer(a.encode(),dtype=np.uint8)==np.frombuffer(b.encode(),dtype=np.uint8)))
                for a,b in itertools.combinations(sequences,2) if len(a)==len(b)]
    return {'exact_unique':len(set(sequences)), 'aligned_identity':identities,
            'fourmer_entropy_bits':float(-(probs*np.log2(probs)).sum())}


def budget_curve(rows, horizon):
    elapsed, seen = 0., set()
    xs, ys = [0.], [0]
    for row in rows:
        if not row.get('budget_eligible',True):
            raise ValueError('存在耗时不完整的中断尝试，不能给出完整固定预算曲线')
        elapsed += row.get('budget_total_s',row['total_s'])
        if elapsed > horizon:
            break
        if row['qualified']:
            seen.add(row['sequence'])
        xs.append(elapsed)
        ys.append(len(seen))
    if xs[-1]<horizon:
        xs.append(horizon)
        ys.append(ys[-1])
    return {'seconds':xs,'qualified_unique':ys}


def summarize(pairs):
    native, spec = zip(*pairs)
    ratios=np.array([n['total_s']/s['total_s'] for n,s in pairs])
    clusters=[n['seed'] for n,s in pairs] if all('seed' in n for n,s in pairs) else None
    out={'speedup_median':bootstrap(ratios,np.median,clusters=clusters),
         'speedup_geometric_mean':bootstrap(np.log(ratios),clusters=clusters),
         'checker_auc_delta':bootstrap([s['checker']['checker_auc']-n['checker']['checker_auc'] for n,s in pairs],clusters=clusters),
         'guide_auc_delta':bootstrap([s['guide']['ensemble_auc']-n['guide']['ensemble_auc'] for n,s in pairs],clusters=clusters)}
    gm=out['speedup_geometric_mean']
    gm['estimate']=float(np.exp(gm['estimate']))
    gm['ci95']=np.exp(gm['ci95']).tolist()
    horizon=min(sum(r.get('budget_total_s',r['total_s']) for r in rows) for rows in (native,spec))
    out['common_budget_seconds']=horizon
    out['arms']={}
    for backend,rows in [('native',native),('evspark',spec)]:
        out['arms'][backend]={'mean_checker_auc':float(np.mean([r['checker']['checker_auc'] for r in rows])),
                              'diversity':diversity([r['sequence'] for r in rows]),
                              'qualified_diversity':diversity([r['sequence'] for r in rows if r['qualified']]),
                              'budget_curve':budget_curve(rows,horizon)}
        arm=out['arms'][backend]
        for source,destination in [('stages_s','mean_stage_seconds'),('loading_s','mean_loading_seconds'),('warmup_s','mean_warmup_seconds')]:
            names=sorted(set().union(*(r.get(source,{}) for r in rows)) - {'recovery'})
            arm[destination]={name:float(np.mean([r.get(source,{}).get(name,0.) for r in rows])) for name in names}
        for name in ('peak_generation_bytes','peak_scorer_bytes'):
            arm[name]=max((r[name] for r in rows if name in r),default=None)
        arm['allocated_gpu_seconds']=sum(r['allocated_gpu_seconds'] for r in rows) if all('allocated_gpu_seconds' in r for r in rows) else None
    out['by_pattern']={}
    for name in sorted({a['pattern'] for a,b in pairs if 'pattern' in a}):
        selected=[(a,b) for a,b in pairs if a['pattern']==name]
        out['by_pattern'][name]={'n_pairs':len(selected),
            'speedup_geometric_mean':float(np.exp(np.mean([np.log(a['total_s']/b['total_s']) for a,b in selected]))),
            'mean_checker_auc':{backend:float(np.mean([p[i]['checker']['checker_auc'] for p in selected]))
                                for i,backend in enumerate(('native','evspark'))}}
    return out


def figure(pairs, summary, output, stage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.4),layout='constrained')
    axes=axes.ravel()
    colors={'native':'#597C96','evspark':'#CE7651'}
    rows={b:[pair[i] for pair in pairs] for i,b in enumerate(colors)}
    bottom=np.zeros(2)
    phases=[('prefill','Prefill','#D7DCE2'),('generation','Generation','#527CA0'),
            ('cache','Cache handling','#9AB9BA'),('scoring','Search scoring','#D79969'),
            ('final_check','Final checking','#AA99B7'),('orchestration_and_io','Scheduling / I/O','#B9B19B')]
    for name,label,color in phases:
        values=[np.mean([r['stages_s'].get(name,0) for r in rows[b]])/60 for b in colors]
        axes[0].bar(list(colors), values, bottom=bottom, label=label,color=color)
        bottom+=values
    axes[0].set(ylabel='Mean full-workflow time (min)',title='A  Time per design')
    axes[0].legend(fontsize=6)
    for b in colors:
        curve=summary['arms'][b]['budget_curve']
        axes[1].step(np.array(curve['seconds'])/3600,curve['qualified_unique'],where='post',color=colors[b],label=b)
    axes[1].set(xlabel='Cumulative elapsed time (h)',ylabel='Qualified unique final designs',title='B  Fixed budget')
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_ylim(bottom=0,top=max(1,max(summary['arms'][b]['budget_curve']['qualified_unique'][-1] for b in colors))*1.12)
    axes[1].legend()
    for pair in pairs:
        axes[2].plot([0,1],[r['checker']['checker_auc'] for r in pair],color='#bbbbbb',alpha=.6)
    for i,b in enumerate(colors):
        axes[2].scatter([i]*len(rows[b]),[r['checker']['checker_auc'] for r in rows[b]],color=colors[b])
    axes[2].axhline(.9,color='gray',ls='--',lw=.7)
    axes[2].set(xticks=[0,1],xticklabels=list(colors),xlim=(-.5,1.5),ylim=(0,1),ylabel='Final Borzoi checker AUROC',title='C  Predicted quality')
    for i,b in enumerate(colors):
        d=summary['arms'][b]['diversity']
        axes[3].scatter([i]*len(d['aligned_identity']),d['aligned_identity'],alpha=.35,color=colors[b])
        axes[3].text(i,.98,f"unique={d['exact_unique']}\nH4={d['fourmer_entropy_bits']:.2f}",ha='center',va='top',fontsize=7)
    axes[3].set(xticks=[0,1],xticklabels=list(colors),xlim=(-.5,1.5),ylim=(0,1),ylabel='Pairwise positionwise identity',title='D  Diversity')
    title={'pilot':'Pilot','formal':'Main experiment','long_validation':'Long designs'}.get(stage,stage)
    fig.suptitle(f"{title}: regulatory DNA design\n{len(pairs)} pairs, {summary['speedup_median']['n_seed_clusters']} seed groups",fontsize=10)
    fig.savefig(str(output)+'.png',dpi=300)
    fig.savefig(str(output)+'.pdf')
    plt.close(fig)


def markdown_report(root,stage,pairs,summary,gate=None):
    """报告只从当前阶段完整配对数据生成；pilot 与正式结果分开。"""
    def estimate(value):
        return f"{value['estimate']:.4f}（95% CI {value['ci95'][0]:.4f}–{value['ci95'][1]:.4f}）"
    lines=[f'# Phase 5 {stage} 实测报告','',
           f"完整配对设计数：{len(pairs)}，种子组数：{summary['speedup_median']['n_seed_clusters']}。统计量按整次设计计算；置信区间将同一种子复用到的各图案成组配对重采样，保留跨图案的相关性。", '',
           '| 指标 | 实测 |','| --- | --- |',
           f"| 耗时比 native/EvSpark 中位数 | {estimate(summary['speedup_median'])} |",
           f"| 耗时比几何均值 | {estimate(summary['speedup_geometric_mean'])} |",
           f"| 检查模型 AUROC 差 EvSpark−native | {estimate(summary['checker_auc_delta'])} |",
           f"| 搜索集成 AUROC 差 EvSpark−native | {estimate(summary['guide_auc_delta'])} |",'']
    if summary['speedup_median']['n_seed_clusters']==2:
        lines+=['仅有两个种子组，所列 bootstrap 区间为探索性结果，不能据此宣称总体收益稳定或质量等价；应结合全部逐设计结果阅读。','']
    if gate is not None:
        lines += ['投入门：'+('通过，达到预设投入标准。' if gate['passed'] else '未通过，按计划停止扩大。'),'',
                  '| 门 | 是否满足 |','| --- | --- |']
        lines += [f'| {k} | {v} |' for k,v in gate['checks'].items()]
        lines += ['', 'pilot 数据不进入正式实验统计；投入门不是统计等价性判据。','']
        scope=root/'execution_scope.json'
        if scope.exists() and json.loads(scope.read_text()).get('through')=='pilot':
            lines += ['用户已将本轮范围收窄为 pilot：无论投入门是否通过，均不启动正式或长设计实验。本报告为小规模探索性应用证据。','']
    lines += ['| 后端 | 检查 AUROC 均值 | 全部输出精确去重数 | 合格输出精确去重数 | 4-mer 熵 |',
              '| --- | --- | --- | --- | --- |']
    for backend,values in summary['arms'].items():
        lines.append(f"| {backend} | {values['mean_checker_auc']:.4f} | {values['diversity']['exact_unique']} | "
                     f"{values['qualified_diversity']['exact_unique']} | {values['diversity']['fourmer_entropy_bits']:.4f} |")
    lines+=['','## 全部逐设计结果','',
            '| 图案 | 种子 | 后端 | 完整分钟 | 搜索 AUROC | 检查 AUROC | 合格 |',
            '| --- | --- | --- | --- | --- | --- | --- |']
    for pair in pairs:
        for row in pair:
            lines.append(f"| {row['pattern']} | {row['seed']} | {row['backend']} | {row['total_s']/60:.3f} | "
                         f"{row['guide']['ensemble_auc']:.4f} | {row['checker']['checker_auc']:.4f} | {row['qualified']} |")
    lines+=['','## 耗时分解与瓶颈','',
            '| 阶段 | Native 平均秒数 | EvSpark 平均秒数 |','| --- | --- | --- |']
    phases=sorted(set().union(*(v['mean_stage_seconds'] for v in summary['arms'].values())))
    for phase in phases:
        lines.append(f"| {phase} | {summary['arms']['native']['mean_stage_seconds'].get(phase,0):.3f} | "
                     f"{summary['arms']['evspark']['mean_stage_seconds'].get(phase,0):.3f} |")
    costs=summary['arms']['evspark']['mean_stage_seconds']
    if costs:
        largest=max(costs,key=costs.get)
        lines+=['',f'EvSpark 最大计时项为 `{largest}`，占其平均完整耗时的 {100*costs[largest]/sum(costs.values()):.1f}%。这是实测成本分解，不包含模型加载和预热。']
    lines+=['','| 后端 | 生成峰值 GiB | 评分峰值 GiB | 占用 GPU-hours | 平均加载秒数 | 平均预热秒数 |',
            '| --- | --- | --- | --- | --- | --- |']
    for backend,v in summary['arms'].items():
        lines.append(f"| {backend} | {v['peak_generation_bytes']/2**30:.3f} | {v['peak_scorer_bytes']/2**30:.3f} | "
                     f"{v['allocated_gpu_seconds']/3600:.3f} | {sum(v['mean_loading_seconds'].values()):.3f} | {sum(v['mean_warmup_seconds'].values()):.3f} |")
    lines+=['','GPU-hours 为两张卡在完整设计期间的占用时长之和；CUDA event 区间另存原始结果，不解释成纯内核忙碌时间。','',
            '## 分图案描述性结果','',
            '| 图案 | 配对设计数 | 耗时比几何均值 | Native 检查 AUROC 均值 | EvSpark 检查 AUROC 均值 |',
            '| --- | --- | --- | --- | --- |']
    for name,v in summary['by_pattern'].items():
        lines.append(f"| {name} | {v['n_pairs']} | {v['speedup_geometric_mean']:.4f} | "
                     f"{v['mean_checker_auc']['native']:.4f} | {v['mean_checker_auc']['evspark']:.4f} |")
    lines += ['', f"公共固定预算范围：0–{summary['common_budget_seconds']:.3f} 秒，仅累计实际完成且合格的最终第一名序列。",'',
              f'技术汇总图：`{stage}_main_figure.png` / `.pdf`；完整统计：`{stage}_summary.json`。','',
              '“合格”仅指两个预定预测 AUROC 均≥0.90，不表示实验功能验证。检查模型未参与搜索，不主张其训练数据独立。',
              '序列相似度为等长输出逐位置的碱基一致率；4-mer 熵由各臂全部最终输出的合并频数计算。二者均为描述性指标。',
              '加载/预热另列；主要耗时比较排除有断点恢复的运行，失败重试开销纳入固定预算。质量差异不显著不能作为等价证据。','',
              '## 原始结果','']
    if stage=='pilot' and (root/'pilot_case_study.pdf').exists():
        lines[-2:-2]=['六面板 case study 主图：`pilot_case_study.png` / `.pdf`；逐设计表：`pilot_designs.tsv`；全部最终序列：`pilot_final_sequences.fasta`；原始轨迹审计：`pilot_audit.json`。','']
    for pair in pairs:
        for row in pair:
            lines.append(f"- `{row['source']}`：batch={row['batch']}，完整耗时={row['total_s']:.3f} 秒。")
    (root/f'{stage}_report.md').write_text('\n'.join(lines)+'\n')


def markdown_calibration(root,stage,records,selection):
    """公开完整 batch 选择依据；单个校准种子的结果不承担最终性能推断。"""
    lines=[f'# Phase 5 {stage} 完整工作流校准','',
           '两臂分别按完整工作流最短耗时选择 batch；生成之外的评分、prefill、缓存和最终检查均计入。',
           '每个 batch 使用独立于 pilot/正式实验的同一校准种子。表中质量只用于披露，不参与 batch 选择；这些运行不进入主要结果统计。','',
           '| 后端 | Batch | 状态 | 完整耗时 (min) | 生成显存 (GiB) | 评分显存 (GiB) | 最终检查 AUROC | 选中 |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in sorted(records,key=lambda r:(r['backend'],r['batch'])):
        chosen=row['batch']==selection['selection'][row['backend']]['batch']
        if row['status']=='complete':
            values=[f"{row['total_s']/60:.3f}",f"{row['peak_generation_bytes']/2**30:.3f}",
                    f"{row['peak_scorer_bytes']/2**30:.3f}",f"{row['checker']['checker_auc']:.4f}"]
        else:values=['—']*4
        lines.append('| '+' | '.join([row['backend'],str(row['batch']),row['status'],*values,'是' if chosen else '否'])+' |')
    lines+=['','设备组合：`'+json.dumps(selection['devices'],ensure_ascii=False)+'`。显存为峰值 allocated memory；模型加载和预热另存各次 result.json。','',
            '## 原始记录','']
    for row in records:
        lines.append(f"- `{row['source']}`"+(f"：{row['error']}" if row.get('error') else ''))
    (root/f'{stage}_report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='benchmarks/phase5')
    p.add_argument('--stage',choices=[*CALIBRATION_FOR.values(),*CALIBRATION_FOR],required=True)
    args=p.parse_args()
    root=Path(args.root)
    records=load_results(root,args.stage)
    if args.stage in CALIBRATION_FOR.values():
        result=select_batches(records)
        out=root/selection_filename(args.stage)
        if out.exists() and json.loads(out.read_text())!=result:
            raise ValueError('已冻结 batch 选择，禁止静默覆盖')
        atomic_json(out,result)
        markdown_calibration(root,args.stage,records,result)
        return
    selection=json.loads((root/selection_filename(CALIBRATION_FOR[args.stage])).read_text())
    pairs=paired(records,args.stage,selection)
    summary=summarize(pairs)
    gate=None
    if args.stage=='pilot':
        correctness=json.loads((root/'generation_validation_40k.json').read_text())
        checks={'generation_correctness':correctness.get('correctness_passed') is True,
                'native_quality':summary['arms']['native']['mean_checker_auc']>=.8,
                'evspark_quality':summary['arms']['evspark']['mean_checker_auc']>=.8,
                'speedup':summary['speedup_median']['estimate']>=1.25}
        # 评分器上游样例核验须另外显式通过，不能由 shape smoke 冒充。
        scorer_validation=root/'scorer_validation.json'
        checks['scorer_correctness']=scorer_validation.exists() and json.loads(scorer_validation.read_text()).get('status')=='passed'
        for backend in ('native_b8','evspark_b4'):
            stress=root/f'branch_stress_{backend}.json'
            if stress.exists():
                checks[f'branch_stress_{backend}']=json.loads(stress.read_text()).get('status')=='passed'
        gate={'passed':all(checks.values()),'checks':checks,'summary':summary}
        atomic_json(root/'pilot_gate.json',gate)
    atomic_json(root/f'{args.stage}_summary.json',summary)
    figure(pairs,summary,root/f'{args.stage}_main_figure',args.stage)
    markdown_report(root,args.stage,pairs,summary,gate)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
