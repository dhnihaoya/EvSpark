"""持久执行下载、评分预检和实验队列；各阶段状态落盘，失败即停。"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .run import atomic_json
from .prepare import received_bytes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scorer-python',default='/tmp/evspark_phase5_env/bin/python')
    p.add_argument('--download-workers',type=int,default=4)
    p.add_argument('--through',choices=['calibration','pilot','all'],default='all')
    p.add_argument('--application-only',action='store_true',help='同一环境恢复队列，复用已经通过的评分验证')
    p.add_argument('--wait-pid',type=int,help='等待仍在计算的本项目设计进程退出后接管队列')
    p.add_argument('--wait-result',type=Path,help='被接管进程的结果，必须完整成功才继续')
    args=p.parse_args()
    if args.wait_pid and (not args.application_only or args.wait_result is None):
        p.error('--wait-pid 要求 --application-only 与 --wait-result')
    project=Path(__file__).resolve().parents[2]
    root=Path('benchmarks/phase5')
    root.mkdir(parents=True,exist_ok=True)
    logs=Path('logs/phase5'); logs.mkdir(parents=True,exist_ok=True)
    lock=open(root/'pipeline.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]),PYTHONUNBUFFERED='1')
    state={'pid':os.getpid(),'started_unix':time.time(),'through':args.through,'status':'running'}
    def save():
        state['updated_unix']=time.time()
        atomic_json(root/'pipeline_status.json',state)
    def execute(stage,command):
        state.update(stage=stage,log=str(logs/f'{stage}.log'));save()
        print(json.dumps(state,ensure_ascii=False),flush=True)
        with open(state['log'],'a') as log:
            process=subprocess.Popen(command,env=env,cwd=project,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=process.pid
            while True:
                try:
                    code=process.wait(timeout=30)
                    state['child_returncode']=code;save()
                    return code
                except subprocess.TimeoutExpired:
                    if stage=='prepare_pipeline':
                        manifest=json.loads(Path('data/phase5_assets/manifest.json').read_text())
                        progress=[]
                        cache=Path(os.environ.get('EVSPARK_PHASE5_DOWNLOAD_CACHE',str(Path.home()/'.cache/evspark/phase5_downloads')))
                        for repo,model in manifest['models'].items():
                            for name,meta in model['remote'].items():
                                if not meta.get('lfs'):
                                    continue
                                total=meta['size'];oid=meta['lfs']['oid'].removeprefix('sha256:')
                                part=cache/oid/(name+'.part')
                                done=name in model['files']
                                received=total if done else received_bytes(part,total)
                                progress.append({'model':repo,'received_bytes':received,'total_bytes':total,'verified':done})
                        state['downloads']=progress
                    save()
    try:
        checked=json.loads((root/'generation_validation_40k.json').read_text())
        if checked.get('correctness_passed') is not True or not checked.get('native_optimized_equal'):
            raise RuntimeError('真实 7B 生成/优化 native 验证未通过，不能启动应用队列')
        if args.wait_pid:
            proc=Path(f'/proc/{args.wait_pid}')
            if proc.exists() and b'regulatory_design.run' not in (proc/'cmdline').read_bytes():
                raise RuntimeError('接管 PID 不是本项目的设计进程')
            state.update(stage='finishing_existing_design',existing_pid=args.wait_pid,existing_result=str(args.wait_result))
            while True:
                try:
                    running=(proc/'stat').read_text().split(') ',1)[1][0]!='Z'
                except FileNotFoundError:
                    running=False
                if not running:break
                save();time.sleep(10)
            result=json.loads(args.wait_result.read_text())
            if result.get('status')!='complete' or not result.get('timing_eligible'):
                raise RuntimeError('待接管的设计未完整成功，保留结果并停止')
            state['existing_design_complete']=True;save()
        if not args.application_only:
            if execute('prepare_pipeline',[sys.executable,'-m','regulatory_design.prepare','--workers',str(args.download_workers)]):
                raise RuntimeError('公开资产下载/校验失败；可从保留的 .part 重启')
        # 完成时刷新终态，避免最后一个权重落在两个 30 秒心跳之间而一直显示未完成。
        manifest=json.loads(Path('data/phase5_assets/manifest.json').read_text())
        for progress in state.get('downloads',[]):
            model=manifest['models'][progress['model']]
            progress['verified']=set(model['files'])==set(model['remote'])
            if progress['verified']:progress['received_bytes']=progress['total_bytes']
        save()
        gpu=1
        if not args.application_only:
            code=execute('scorer_validation',[args.scorer_python,'-m','regulatory_design.validate_scoring','--gpu','1'])
            if code:
                validation=json.loads((root/'scorer_validation.json').read_text())
                if 'out of memory' not in validation.get('error','').lower():
                    raise RuntimeError('评分验证失败，详见 scorer_validation.json')
                atomic_json(root/'scorer_validation_gpu1_oom.json',validation)
                gpu=2
                if execute('scorer_validation_gpu2',[args.scorer_python,'-m','regulatory_design.validate_scoring','--gpu','2']):
                    raise RuntimeError('GPU2 评分验证失败，停止实验')
        validation=json.loads((root/'scorer_validation.json').read_text())
        if validation.get('status')!='passed':
            raise RuntimeError('评分验证未放行')
        if args.application_only:gpu=validation['gpu']
        state['scorer_gpu']=gpu;save()
        if execute('application_queue',[sys.executable,'-m','regulatory_design.launch','--python',sys.executable,
                                       '--scorer-python',args.scorer_python,'--scorer-gpu',str(gpu),'--through',args.through]):
            raise RuntimeError('应用队列失败，详见 application_queue.log 与对应设计 run.log')
        queue_path=root/'queue_status.json'
        state['status']=json.loads(queue_path.read_text())['status'] if queue_path.exists() else f'completed_through_{args.through}'
        save()
    except Exception as exc:
        state.update(status='failed',error=repr(exc));save()
        raise


if __name__=='__main__':
    main()
