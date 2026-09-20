"""按冻结顺序执行校准、pilot 和过门后的正式实验；可重启，拒绝并发重复队列。"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from .protocol import PROTOCOL, CALIBRATION_FOR, selection_filename
from .run import atomic_json, implementation_hash


def execution_limit(root, requested):
    """用户后续收窄的范围持久生效，旧 --through all 命令不能意外扩大。"""
    path=Path(root)/'execution_scope.json'
    if not path.exists():
        return requested
    limit=json.loads(path.read_text())['through']
    order=('calibration','pilot','all')
    if limit not in order:
        raise ValueError('执行范围无效，停止以免意外扩大实验')
    return order[min(order.index(requested),order.index(limit))]


def archive_attempt(output,old):
    """完整保留被重跑的尝试，避免新轨迹覆盖旧候选；结果原子移入预算索引。"""
    output=Path(output)
    attempts=output/'attempts'
    attempts.mkdir(exist_ok=True)
    index=len(list(attempts.glob('*.json')))
    evidence=attempts/f'{index:03d}_files'
    evidence.mkdir(exist_ok=True)
    for name in ('trajectory.jsonl','progress.json','failed_round.json','final.fasta','run.log','scorer.log'):
        path=output/name
        if path.exists():path.replace(evidence/name)
    if old['status']=='running' or (old['status']=='complete' and not old.get('timing_eligible')):
        old=dict(old,elapsed_incomplete=True)
    atomic_json(output/'result.json',old)
    (output/'result.json').replace(attempts/f'{index:03d}.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--scorer-python',default='/tmp/evspark_phase5_env/bin/python')
    p.add_argument('--root',default='benchmarks/phase5')
    p.add_argument('--scorer-gpu',type=int,default=1)
    p.add_argument('--through',choices=['calibration','pilot','all'],default='all')
    args=p.parse_args()
    project=Path(__file__).resolve().parents[2]
    root=Path(args.root)
    root.mkdir(parents=True,exist_ok=True)
    args.through=execution_limit(root,args.through)
    lock=open(root/'queue.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(Path(__file__).resolve().parents[1]),
             PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sleep_status=subprocess.run(['systemctl','is-enabled','sleep.target'],capture_output=True,text=True)
    if sleep_status.stdout.strip()!='masked':
        raise RuntimeError('sleep.target 未 mask；按项目约定先处理禁止挂起配置')
    validation=json.loads((root/'generation_validation_40k.json').read_text())
    if validation.get('correctness_passed') is not True:
        raise RuntimeError('生成正确性尚未放行')
    for name in ('branch_stress_native_b8.json','branch_stress_evspark_b4.json'):
        stress=root/name
        if stress.exists() and json.loads(stress.read_text()).get('status')!='passed':
            raise RuntimeError(f'已有分支压力验证未通过：{name}')
    scorer=json.loads((root/'scorer_validation.json').read_text())
    if scorer.get('status')!='passed' or scorer['gpu']!=args.scorer_gpu:
        raise RuntimeError('所选 GPU 的评分验证尚未通过')
    score_batch=scorer['selected_score_batch']

    def job(stage,backend,batch,pattern,seed,length):
        name=f'{stage}_{pattern}_{seed}_{backend}_b{batch}'
        output=root/name
        output.mkdir(parents=True,exist_ok=True)
        result_path=output/'result.json'
        if result_path.exists():
            old=json.loads(result_path.read_text())
            if old.get('implementation_sha256')!=implementation_hash() or old.get('score_batch')!=score_batch:
                raise RuntimeError(f'{name} 的代码或评分 batch 已变化，禁止跳过或混入原队列')
            if old['status']=='complete' and old.get('timing_eligible'):
                print(f'已完成，跳过 {name}',flush=True)
                return
            if stage in CALIBRATION_FOR.values() and old['status']=='failed' and 'out of memory' in old.get('error','').lower():
                print(f'已记录 OOM，跳过 {name}',flush=True)
                return
            # 保留旧失败/中断记录；重跑完整设计才能进入计时比较。
            archive_attempt(output,old)
        command=[args.python,'-u','-m','regulatory_design.run','--backend',backend,'--batch',str(batch),
                 '--pattern',pattern,'--seed',str(seed),'--length',str(length),'--stage',stage,
                 '--scorer-python',args.scorer_python,'--scorer-gpu',str(args.scorer_gpu),'--output',str(output)]
        command.extend(['--score-batch',str(score_batch)])
        print('执行 '+name,flush=True)
        with open(output/'run.log','a') as log:
            completed=subprocess.run(command,env=env,cwd=project,stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:
            failure=json.loads(result_path.read_text()) if result_path.exists() else {}
            if stage in CALIBRATION_FOR.values() and 'out of memory' in failure.get('error','').lower():
                return
            raise RuntimeError(f'{name} 失败；详见 {output}/run.log')

    def report(stage):
        with open(root/f'{stage}_report.log','w') as log:
            subprocess.run([args.python,'-m','regulatory_design.report','--root',str(root),'--stage',stage],
                           env=env,cwd=project,stdout=log,stderr=subprocess.STDOUT,check=True)

    for stage in ('pilot','formal','long_validation'):
        calibration=CALIBRATION_FOR[stage]
        cal=PROTOCOL[calibration]
        for batch in (1,2,4,8):
            for backend in ('native','evspark'):
                if batch in PROTOCOL[backend+'_batches']:
                    job(calibration,backend,batch,cal['pattern'],cal['seed'],cal['length'])
        report(calibration)
        if args.through=='calibration':
            return
        selection=json.loads((root/selection_filename(calibration)).read_text())['selection']
        cfg=PROTOCOL[stage]
        for pattern in cfg['patterns']:
            for i,seed in enumerate(cfg['seeds']):
                # 成对执行且交替臂顺序，固定预算仍按 pattern × seed 的预定顺序汇总。
                arms=('native','evspark') if i%2==0 else ('evspark','native')
                for backend in arms:
                    job(stage,backend,selection[backend]['batch'],pattern,seed,cfg['length'])
        report(stage)
        if stage=='pilot':
            gate=json.loads((root/'pilot_gate.json').read_text())
            if not gate['passed']:
                atomic_json(root/'queue_status.json',{'status':'stopped_at_pilot_gate','checks':gate['checks']})
                print('pilot 未过门，按计划停止扩大。',flush=True)
                return
            if args.through=='pilot':
                atomic_json(root/'queue_status.json',{'status':'completed_through_pilot',
                            'through':'pilot','pilot_gate_passed':True})
                print('pilot 报告已生成，按本轮执行范围收尾，不扩大实验。',flush=True)
                return
    with open(root/'paper_materialization.log','w') as log:
        subprocess.run([args.python,'-m','regulatory_design.paper','--root',str(root),'--compile'],
                       env=env,cwd=project,stdout=log,stderr=subprocess.STDOUT,check=True)
    # 公开复现包没有主仓历史数字脚本；主仓入稿后同时核验全部已有数字。
    audit_script=Path('scripts/check_paper_numbers.py')
    if audit_script.exists():
        with open(root/'all_paper_numbers.log','w') as log:
            subprocess.run([args.python,str(audit_script)],env=env,cwd=project,stdout=log,stderr=subprocess.STDOUT,check=True)
    atomic_json(root/'queue_status.json',{'status':'completed_all_experiments_and_paper_checks'})


if __name__=='__main__':
    main()
