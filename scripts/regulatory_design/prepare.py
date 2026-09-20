"""下载并锁定公开模型、轨道和 mm39 背景；无 HF 令牌依赖。"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def remote_json(url):
    result = subprocess.run(['curl', '--noproxy', '*', '--fail', '--silent', '--show-error', '--location',
                             '--retry', '3', '--max-time', '90', url], check=True, capture_output=True)
    return json.loads(result.stdout)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def received_bytes(part,total):
    """分段下载不能用稀疏文件表观长度表示进度；读取 aria2 的实际完成字节。"""
    part=Path(part)
    control=part.with_name(part.name+'.aria2')
    if control.exists():
        log=part.parent/'download.log'
        if log.exists():
            with open(log,'rb') as handle:
                handle.seek(max(0,log.stat().st_size-65536))
                matches=re.findall(rb'(\d+)(?:B)?/(\d+)(?:B)?\(\d+%\)',handle.read())
            matches=[int(a) for a,b in matches if int(b)==total]
            if matches:return min(total,matches[-1])
        return 0
    return min(total,part.stat().st_size) if part.exists() else 0


def download(url, dest, expected=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        # NTFS 上 curl 进度日志曾卡在内核写锁；下载、日志、锁均置于原生文件系统。
        cache=Path(os.environ.get('EVSPARK_PHASE5_DOWNLOAD_CACHE',str(Path.home()/'.cache/evspark/phase5_downloads')))
        identity=expected or hashlib.sha256(url.encode()).hexdigest()
        folder=cache/identity
        folder.mkdir(parents=True,exist_ok=True)
        complete=folder/dest.name
        part=folder/(dest.name+'.part')
        with open(folder/'download.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            legacy=dest.with_name(dest.name+'.part')
            if not complete.exists():
                if not part.exists() and legacy.exists():
                    shutil.copyfile(legacy,part)
                with open(folder/'download.log','a') as log:
                    aria2=os.environ.get('EVSPARK_PHASE5_ARIA2')
                    if aria2:
                        command=[aria2,'--no-conf','--no-netrc=true','--all-proxy=','--http-proxy=','--https-proxy=',
                                 '--continue=true','--allow-overwrite=true','--auto-file-renaming=false','--file-allocation=none',
                                 '--max-connection-per-server=4','--split=4','--min-split-size=16M',
                                 '--max-tries=8','--retry-wait=3','--connect-timeout=30','--timeout=60',
                                 '--human-readable=false','--enable-color=false','--truncate-console-readout=false','--summary-interval=10',
                                 '--dir='+str(folder),'--out='+part.name]
                        if expected:command += ['--checksum=sha-256='+expected,'--check-integrity=true']
                        subprocess.run(command+[url],check=True,stdout=log,stderr=log)
                    else:
                        if part.with_name(part.name+'.aria2').exists():
                            raise ValueError('该断点含未完成分段，须保留 EVSPARK_PHASE5_ARIA2；不能用顺序 curl 误读稀疏文件')
                        for attempt in range(7):
                            try:
                                # curl 内部 retry 会退回启动时的偏移；外层重启才能沿最新断点续传。
                                subprocess.run(['curl', '--noproxy', '*', '--fail', '--location',
                                                '--max-time','7200','--speed-limit', '1024', '--speed-time', '60', '--connect-timeout', '30',
                                                '--continue-at', '-', '--output', str(part), url], check=True,stdout=log,stderr=log)
                                break
                            except subprocess.CalledProcessError:
                                if attempt==6:raise
                                time.sleep(3)
                if expected and digest(part)!=expected:
                    raise ValueError(f'{part} 下载校验失败，未发布到资产目录')
                part.replace(complete)
            if expected and digest(complete)!=expected:
                raise ValueError(f'{complete} 缓存校验失败')
            # 模型加载和 manifest 保持稳定的项目内路径，实体保存在本机原生缓存。
            if not dest.exists():
                dest.symlink_to(complete.resolve())
    sha = digest(dest)
    if expected and sha != expected:
        raise ValueError(f'{dest} 校验失败：{sha} != {expected}')
    return {'path': str(dest), 'sha256': sha, 'bytes': dest.stat().st_size, 'url': url}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/phase5_assets'))
    p.add_argument('--endpoint', default='https://hf-mirror.com')
    p.add_argument('--lock',type=Path,default=Path(__file__).with_name('assets.lock.json'),help='发布时冻结的版本/校验和清单')
    p.add_argument('--reference-dir',type=Path,default=Path(__file__).with_name('reference'),help='随复现包发布的轨道及参考背景')
    p.add_argument('--workers',type=int,default=1,help='并行下载文件数；本机实测单连接更稳定')
    args = p.parse_args()
    # 本机继承的代理对 GitHub/UCSC 超时；公开资产直连已验证可用。
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'manifest.json'
    locked=json.loads(args.lock.read_text()) if args.lock.exists() else None
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else (locked or {'models': {}, 'files': {}})
    if locked and any(manifest['models'].get(repo,{}).get('revision')!=m['revision'] for repo,m in locked['models'].items()):
        raise ValueError('本地 manifest 与发布的模型版本锁不一致')

    def save():
        tmp = manifest_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
        tmp.replace(manifest_path)

    def bundled(name):
        src=args.reference_dir/name
        dest=root/name
        if src.exists() and not dest.exists():
            # UCSC API JSON 含下载时间；复现包提供当时原始响应，避免把时间戳变化误判为序列变化。
            shutil.copyfile(src,dest)

    # 先获取小资产；模型逐个记录，支持中断后继续。
    tables = {
        'enformer_targets': 'https://raw.githubusercontent.com/calico/basenji/0.5/manuscripts/cross2020/targets_mouse.txt',
        'borzoi_targets': 'https://raw.githubusercontent.com/calico/borzoi/main/examples/targets_mouse.txt',
    }
    for key, url in tables.items():
        bundled(key+'.tsv')
        expected=manifest['files'].get(key,{}).get('sha256')
        manifest['files'][key] = download(url, root / (key + '.tsv'),expected)
        save()
    start, end = 52051928, 52123468
    for key, lo, hi in [('upstream', start - 163840, start), ('downstream', end, end + 360448)]:
        bundled(key+'.json')
        url = f'https://api.genome.ucsc.edu/getData/sequence?genome=mm39;chrom=chrX;start={lo};end={hi}'
        entry = download(url, root / (key + '.json'),manifest['files'].get(key,{}).get('sha256'))
        record = json.loads(Path(entry['path']).read_text())
        if len(record['dna']) != hi - lo or record['start'] != lo or record['end'] != hi:
            raise ValueError('UCSC 序列坐标/长度不符')
        entry.update(assembly='mm39', chrom='chrX', start=lo, end=hi)
        manifest['files'][key] = entry
        save()

    fixtures={
        'enformer_test_sample.pt':'https://raw.githubusercontent.com/lucidrains/enformer-pytorch/main/data/test-sample.pt',
        'test_pretrained.py':'https://raw.githubusercontent.com/lucidrains/enformer-pytorch/main/test_pretrained.py',
        'borzoi_example.ipynb':'https://raw.githubusercontent.com/johahi/borzoi-pytorch/main/notebooks/pytorch_borzoi_example.ipynb',
        'borzoi_wt_seq.npy':'https://raw.githubusercontent.com/johahi/borzoi-pytorch/refs/heads/main/wt_seq.npy',
    }
    for name,url in fixtures.items():
        key='validation_'+name
        manifest['files'][key]=download(url,root/'upstream_validation'/name,manifest['files'].get(key,{}).get('sha256'))
        save()

    repos = ['EleutherAI/enformer-official-rough'] + [f'johahi/flashzoi-replicate-{i}' for i in range(4)] + ['johahi/borzoi-replicate-0-mouse']
    jobs=[]
    for repo in repos:
        if repo not in manifest['models']:
            info = remote_json(args.endpoint + '/api/models/' + repo)
            revision = info['sha']
            files = remote_json(args.endpoint + '/api/models/' + repo + '/tree/' + revision + '?recursive=true')
            by_name = {r['path']: r for r in files if r['type'] == 'file'}
            weight = 'model.safetensors' if 'model.safetensors' in by_name else 'pytorch_model.bin'
            chosen = ['config.json', weight]
            manifest['models'][repo] = {'revision': revision, 'files': {}, 'remote': {name: by_name[name] for name in chosen}}
            save()
        model = manifest['models'][repo]
        for name, meta in model['remote'].items():
            url = args.endpoint + '/' + repo + '/resolve/' + model['revision'] + '/' + name
            expected = meta.get('lfs', {}).get('oid')
            if not expected:
                expected=model['files'].get(name,{}).get('sha256')
            if expected and expected.startswith('sha256:'):
                expected = expected[7:]
            jobs.append((repo,name,url,root/'models'/repo.split('/')[-1]/name,expected))
    # 只有主线程写 manifest，避免并发下载覆盖已经完成的记录。
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(download,url,dest,expected):(repo,name) for repo,name,url,dest,expected in jobs}
        for future in as_completed(pending):
            repo,name=pending[future]
            manifest['models'][repo]['files'][name]=future.result()
            print(f'完成 {repo}/{name}',flush=True)
            save()
    print(f'资产齐备：{manifest_path} sha256={digest(manifest_path)}', flush=True)


if __name__ == '__main__':
    main()
