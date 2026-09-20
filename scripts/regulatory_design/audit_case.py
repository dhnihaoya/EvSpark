"""从原始搜索轨迹核验 case study，并导出逐设计表、最终序列和来源校验和。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

from .protocol import PROTOCOL, auc, pattern
from .report import load_results, paired, select_batches, summarize
from .run import atomic_json, candidate_seed


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(2**20),b''):
            digest.update(block)
    return digest.hexdigest()


def audit_design(row):
    """核验所有候选的父分支、种子、长度、全局选择及最终预测统计。"""
    directory=Path(row['source']).parent
    beams={'root':''}
    generated=invalid=0
    trajectory=directory/'trajectory.jsonl'
    records=[json.loads(line) for line in trajectory.read_text().splitlines()]
    assert len(records)==row['length']//128, '搜索轮数不符'
    for number,record in enumerate(records):
        assert record['round']==number and record['generated_bp']==128*(number+1)
        candidates=record['candidates']
        assert len(candidates)==15*len(beams), '独立候选数不符'
        parent_ids=list(beams)
        sequences={}
        for index,candidate in enumerate(candidates):
            rank,local_index=divmod(index,15)
            assert candidate['id']==f'{number}:{rank}:{local_index}'
            assert candidate['parent_id']==parent_ids[rank], '候选父分支错位'
            assert candidate['seed']==candidate_seed(row['seed'],number,rank,local_index)
            assert len(candidate['suffix'])==128
            assert sum(step['consumed'] for step in candidate['sampling_trace'])==128
            sequences[candidate['id']]=beams[candidate['parent_id']]+candidate['suffix']
            invalid+=candidate['score']['loss'] is None
        valid=[i for i,candidate in enumerate(candidates) if candidate['score']['loss'] is not None]
        chosen=sorted(valid,key=lambda i:(candidates[i]['score']['loss'],i))[:2]
        assert len(chosen)==2
        assert record['retained_ids']==[candidates[i]['id'] for i in chosen], '未按全局评分保留两个分支'
        beams={key:sequences[key] for key in record['retained_ids']}
        generated+=len(candidates)
    assert next(iter(beams.values()))==row['sequence'], '最终序列与完整轨迹不符'
    assert candidates[chosen[0]]['score']==row['guide'], '最终搜索评分错位'
    fasta=''.join(line for line in (directory/'final.fasta').read_text().splitlines() if not line.startswith('>'))
    assert fasta==row['sequence']
    guide=row['guide']
    enformer=np.asarray(guide['enformer_profile'])/guide['enformer_normalizer']
    replicates=np.asarray(guide['flashzoi_profiles'])/guide['flashzoi_normalizer']
    flashzoi=replicates.mean(axis=0)-replicates.std(axis=0,ddof=0)
    e_target=pattern(row['pattern'],row['length'],128)
    f_target=pattern(row['pattern'],row['length'],32)
    loss=.5*np.abs(enformer-e_target).sum()+.5*np.abs(flashzoi-f_target).sum()
    ensemble_auc=(auc(e_target,enformer)+auc(f_target,flashzoi))/2
    checker_auc=auc(f_target,row['checker']['checker_profile'])
    assert np.isclose(loss,guide['loss'],rtol=0,atol=1e-10), '最终损失无法复算'
    assert ensemble_auc==guide['ensemble_auc'] and checker_auc==row['checker']['checker_auc']
    assert row['qualified']==(ensemble_auc>=.9 and checker_auc>=.9)
    assert np.isclose(sum(v for k,v in row['stages_s'].items() if k!='recovery'),row['total_s'],rtol=0,atol=1e-6)
    assert row['timing_eligible'] and not row['resumed']
    return {'source':row['source'],'candidate_count':generated,'invalid_candidates':invalid,
            'files':{name:file_hash(directory/name) for name in ('result.json','trajectory.jsonl','final.fasta','protocol.json')}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('benchmarks/phase5'))
    args=parser.parse_args()
    root=args.root
    selection=json.loads((root/'selected_batches.json').read_text())
    calibration=load_results(root,'calibration')
    selected_again=select_batches(calibration)
    assert selected_again['devices']==selection['devices']
    for backend in ('native','evspark'):
        for key in ('batch','total_s'):
            assert selected_again['selection'][backend][key]==selection['selection'][backend][key], '最快 batch 无法复核'
    pairs=paired(load_results(root,'pilot'),'pilot',selection)
    summary=summarize(pairs)
    assert summary==json.loads((root/'pilot_summary.json').read_text()), '汇总统计无法复算'
    rows=[row for pair in pairs for row in pair]
    audit=[audit_design(row) for row in rows]
    fields=['pattern','seed','backend','batch','length','total_s','guide_auc','checker_auc',
            'qualified','peak_generation_bytes','peak_scorer_bytes','sequence_sha256','source']
    with (root/'pilot_designs.tsv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,delimiter='\t')
        writer.writeheader()
        for row in rows:
            output={key:row[key] for key in fields if key in row}
            output.update(guide_auc=row['guide']['ensemble_auc'],checker_auc=row['checker']['checker_auc'],
                          sequence_sha256=hashlib.sha256(row['sequence'].encode()).hexdigest())
            writer.writerow(output)
    with (root/'pilot_final_sequences.fasta').open('w') as handle:
        for row in rows:
            handle.write(f">{row['backend']}_{row['pattern']}_{row['seed']} qualified={row['qualified']}\n")
            sequence=row['sequence']
            handle.write('\n'.join(sequence[i:i+80] for i in range(0,len(sequence),80))+'\n')
    result={'status':'passed','stage':'pilot','n_designs':len(rows),'n_pairs':len(pairs),
            'n_seed_groups':len(PROTOCOL['pilot']['seeds']),
            'checks':['全部搜索轨迹、候选数和种子','全局分支选择、最终序列与 FASTA',
                      '最终预测分数、AUROC 和合格标记','阶段计时总和','全部汇总统计和区间','完整校准的最快 batch 选择'],
            'calibration_sources':{row['source']:file_hash(row['source']) for row in calibration},
            'validation_sources':{name:file_hash(root/name) for name in
                                  ('selected_batches.json','execution_scope.json','generation_validation_40k.json',
                                   'scorer_validation.json','candidate_loop_validation.json') if (root/name).exists()},
            'sources':audit,'exports':{name:file_hash(root/name) for name in
                                      ('pilot_designs.tsv','pilot_final_sequences.fasta','pilot_summary.json')}}
    atomic_json(root/'pilot_audit.json',result)
    print(json.dumps({key:value for key,value in result.items() if key not in ('sources','exports')},ensure_ascii=False))


if __name__=='__main__':
    main()
