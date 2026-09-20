"""应用工作流的坐标、冻结评分与独立候选语义回归。"""
import json
import numpy as np
import pytest
from regulatory_design.protocol import pattern, auc, score_predictions, Background, track_index, freeze


def test_patterns_and_mask():
    assert np.array_equal(pattern('medium', 3072, 128), np.tile([1]*6+[0]*6, 2))
    assert np.array_equal(pattern('short', 3072, 128), np.tile([1]*3+[0]*9, 2))
    assert len(pattern('long', 6144, 32)) == 192
    assert len(pattern('arc', 19968, 128)) == 156
    e, f = np.ones(896), np.ones((4, 6144))
    score = score_predictions(e, f, 'medium', 3072)
    assert score['loss'] == .5*(12+48)
    assert score['ensemble_auc'] == .5
    # 未生成区域只影响归一化，不直接累加损失。
    e2, f2 = e.copy(), f.copy()
    e2[24:] = 0
    f2[:, 96:] = 0
    assert score_predictions(e2, f2, 'medium', 3072) == score


def test_auc_ties_and_single_class():
    assert auc([0, 1, 0, 1], [1, 2, 1, 2]) == 1.
    assert auc([0, 1], [1, 1]) == .5
    assert auc([0, 1], [2, 1]) == 0.
    assert auc([1, 1], [0, 1]) is None


def test_coordinate_replacement(tmp_path):
    (tmp_path/'upstream.json').write_text(json.dumps({'dna':'A'*163840}))
    (tmp_path/'downstream.json').write_text(json.dumps({'dna':'T'*360448}))
    b = Background(tmp_path)
    assert len(b.prompt) == 40960
    for model, left, total in [('enformer',40960,196608),('borzoi',163840,524288)]:
        seq = b.scorer_input('C'*3072, model)
        assert len(seq) == total
        assert seq[left-1:left+2] == 'ACC'
        assert seq[left+3071:left+3074] == 'CTT'


def test_track_positional_index_and_freeze(tmp_path):
    p = tmp_path/'targets.tsv'
    p.write_text('index\tidentifier\n5313\tA\n5324\tENCFF872IES\n')
    assert track_index(p) == 1
    lock = tmp_path/'protocol.json'
    sha = freeze(lock)
    assert freeze(lock) == sha
    lock.write_text('{}')
    with pytest.raises(ValueError):
        freeze(lock)


def test_independent_drafter_contexts():
    torch = pytest.importorskip('torch')
    from evspark.train.drafter import Drafter
    torch.manual_seed(42)
    d = Drafter(d_model=64, n_layers=1, target_hidden=32, gamma=2)
    h = torch.randn(3, 7, 32)
    batched = d.project_context(h, 3)
    assert torch.allclose(batched, torch.cat([d.project_context(row, 1) for row in h]), atol=1e-6)
    changed = h.clone()
    changed[1].add_(3)
    modified = d.project_context(changed, 3)
    assert torch.equal(batched[0], modified[0])
    assert torch.equal(batched[2], modified[2])
    assert not torch.allclose(batched[1], modified[1])


def test_independent_rejection_sampling_joint_law():
    from evspark.specdec.verifier import sample_categorical, verify_round
    # 两行不同 q，各自对同一 p 做拒绝采样；联合分布必须为 p⊗p。
    p = np.array([.15, .35, .5])
    qs = [np.array([.7,.2,.1]), np.array([.1,.7,.2])]
    rngs = [np.random.default_rng(137), np.random.default_rng(982)]
    counts = np.zeros((3,3))
    for _ in range(20000):
        tokens = []
        for q, rng in zip(qs, rngs):
            proposal = sample_categorical(q, rng)
            result = verify_round(np.array([proposal]), q[None], np.stack([p,p]), rng)
            tokens.append(result.tokens[0])
        counts[tuple(tokens)] += 1
    assert np.max(np.abs(counts/counts.sum() - p[:,None]*p[None,:])) < .012


def test_slice_preserves_different_start_offsets():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('slice 经 block 子包引入 vortex，导入需可见 CUDA 驱动')
    from types import SimpleNamespace
    from evspark.specdec.block.slice import ChunkStash, slice_states_to_accept
    ip = {k:SimpleNamespace(seqlen_offset=100, fir_state_dict={}, state_dict={}, fir_inner_state_dict={})
          for k in ('mha','hcl','hcm','hcs')}
    ip['mha'].max_batch_size=2
    ip['mha'].lengths_per_sample=torch.tensor([83,100],dtype=torch.int32)
    ip['mha'].key_value_memory_dict={0:torch.zeros(2,120,2,1,1)}
    ip['hcl'].fir_state_dict[7]=torch.zeros(2,1,2)
    states=torch.arange(2*3*4,dtype=torch.float32).reshape(2,3,4)
    stash=ChunkStash(L0=100,chunk_len=4,s_all={7:states},
                     lengths_before=torch.tensor([83,100],dtype=torch.int32))
    slice_states_to_accept(ip,stash,[2,0])
    assert ip['mha'].lengths_per_sample.tolist()==[86,101]
    assert all(ip[k].seqlen_offset==101 for k in ip)
    assert torch.equal(ip['hcl'].state_dict[7],torch.stack([states[0,:,2],states[1,:,0]]))


def test_budget_never_extrapolates_and_deduplicates():
    from regulatory_design.report import budget_curve, diversity
    rows=[{'total_s':3.,'qualified':True,'sequence':'ACGTACGT'},
          {'total_s':4.,'qualified':True,'sequence':'ACGTACGT'},
          {'total_s':4.,'qualified':True,'sequence':'TTTTTTTT'}]
    curve=budget_curve(rows,9.)
    assert curve=={'seconds':[0.,3.,7.,9.], 'qualified_unique':[0,1,1,1]}
    assert diversity([r['sequence'] for r in rows])['exact_unique']==2


def test_bootstrap_preserves_shared_seed_dependence_across_patterns():
    from regulatory_design.report import bootstrap
    values=np.repeat([1.,2.,3.,4.],3)
    clustered=bootstrap(values,clusters=np.repeat([101,102,103,104],3))
    unclustered=bootstrap(values)
    assert clustered['n_designs']==12 and clustered['n_seed_clusters']==4
    assert clustered['estimate']==unclustered['estimate']==2.5
    assert np.ptp(clustered['ci95'])>np.ptp(unclustered['ci95'])
    with pytest.raises(ValueError,match='不完整分组'):
        bootstrap([1.,2.,3.],clusters=[101,101,102])


def test_calibration_rejects_mixed_implementation_and_nonoom_failure():
    from regulatory_design.report import select_batches
    rows=[]
    for backend,batches in [('native',[1,2,4,8]),('evspark',[1,2,4])]:
        for batch in batches:
            rows.append(dict(backend=backend,batch=batch,status='complete',timing_eligible=True,
                             total_s=100/batch,source=f'{backend}{batch}',devices={'generation':'0','scoring':1},
                             **{k:'fixed' for k in ('protocol_sha256','assets_sha256','checkpoint_sha256',
                                                  'target_sha256','implementation_sha256','score_batch')}))
    selected=select_batches(rows)
    assert selected['selection']['native']['batch']==8
    assert selected['selection']['evspark']['batch']==4
    rows[0]['implementation_sha256']='changed'
    with pytest.raises(ValueError,match='implementation_sha256'):
        select_batches(rows)
    rows[0]['implementation_sha256']='fixed'
    rows[0].update(status='failed',error='shape mismatch')
    with pytest.raises(ValueError,match='非 OOM'):
        select_batches(rows)
    rows[0]['error']='CUDA out of memory'
    assert select_batches(rows)['selection']['native']['batch']==8


def test_asset_download_resumes_on_native_cache_and_checks_before_publish(tmp_path,monkeypatch):
    import hashlib
    from regulatory_design import prepare
    cache=tmp_path/'native_cache'
    monkeypatch.setenv('EVSPARK_PHASE5_DOWNLOAD_CACHE',str(cache))
    destination=tmp_path/'assets'/'model.bin'
    destination.parent.mkdir()
    destination.with_name('model.bin.part').write_bytes(b'prefix')
    expected=hashlib.sha256(b'prefixsuffix').hexdigest()
    def transfer(command,**kwargs):
        part=__import__('pathlib').Path(command[command.index('--output')+1])
        assert part.is_relative_to(cache)
        assert part.read_bytes()==b'prefix'
        with open(part,'ab') as handle:handle.write(b'suffix')
    monkeypatch.setattr(prepare.subprocess,'run',transfer)
    info=prepare.download('https://example.invalid/model.bin',destination,expected)
    assert destination.is_symlink()
    assert destination.read_bytes()==b'prefixsuffix'
    assert info['sha256']==expected
    bad=tmp_path/'assets'/'bad.bin'
    bad.with_name('bad.bin.part').write_bytes(b'prefix')
    with pytest.raises(ValueError,match='下载校验失败'):
        prepare.download('https://example.invalid/bad.bin',bad,'0'*64)
    assert not bad.exists()


def test_sparse_download_progress_and_retry_preserve_latest_offset(tmp_path,monkeypatch):
    import hashlib,subprocess
    from pathlib import Path
    from regulatory_design import prepare
    part=tmp_path/'weights.part'
    part.write_bytes(b'0'*1000)
    part.with_name(part.name+'.aria2').write_bytes(b'control')
    (tmp_path/'download.log').write_text('[#test 200B/1000B(20%) CN:4]\n[#test 330/1000(33%) CN:4]\n')
    assert prepare.received_bytes(part,1000)==330
    monkeypatch.delenv('EVSPARK_PHASE5_ARIA2',raising=False)
    monkeypatch.setenv('EVSPARK_PHASE5_DOWNLOAD_CACHE',str(tmp_path/'cache'))
    monkeypatch.setattr(prepare.time,'sleep',lambda _:None)
    calls=[]
    def transfer(command,**kwargs):
        p=Path(command[command.index('--output')+1]);calls.append(command)
        assert '--retry' not in command
        if len(calls)==1:
            p.write_bytes(b'first')
            raise subprocess.CalledProcessError(18,command)
        assert p.read_bytes()==b'first'
        with open(p,'ab') as handle:handle.write(b'second')
    monkeypatch.setattr(prepare.subprocess,'run',transfer)
    target=tmp_path/'dest.bin'
    prepare.download('https://example.invalid/resume',target,hashlib.sha256(b'firstsecond').hexdigest())
    assert target.read_bytes()==b'firstsecond'
    assert len(calls)==2


def test_paper_gate_fails_closed_and_splice_preserves_other_content(tmp_path):
    from regulatory_design.paper import validated_stages,splice
    (tmp_path/'pilot_gate.json').write_text(json.dumps({'passed':False,'checks':{'speedup':False}}))
    with pytest.raises(ValueError,match='未过预设'):
        validated_stages(tmp_path)
    base='existing result\nANCHOR\nexisting methods'
    fragment='BEGIN\nnew result\nEND\n'
    updated=splice(base,fragment,'BEGIN','END','ANCHOR')
    assert updated.startswith('existing result\n') and updated.endswith('existing methods')
    assert splice(updated,fragment,'BEGIN','END','ANCHOR')==updated
    for bad in ('BEGIN\nANCHOR','END\nANCHOR','END\nBEGIN\nANCHOR','BEGIN\nBEGIN\nEND\nANCHOR'):
        with pytest.raises(ValueError):
            splice(bad,fragment,'BEGIN','END','ANCHOR')


def test_rerun_keeps_failed_trajectory_and_can_replace_resumed_completion(tmp_path):
    from regulatory_design.launch import archive_attempt
    for status in ('running','complete'):
        old={'status':status,'timing_eligible':False}
        (tmp_path/'result.json').write_text(json.dumps(old))
        (tmp_path/'trajectory.jsonl').write_text(status+' original candidates\n')
        (tmp_path/'progress.json').write_text('{"round": 8}')
        archive_attempt(tmp_path,old)
        assert not (tmp_path/'result.json').exists()
        assert not (tmp_path/'trajectory.jsonl').exists()
    first=json.loads((tmp_path/'attempts/000.json').read_text())
    assert first['elapsed_incomplete'] is True
    second=json.loads((tmp_path/'attempts/001.json').read_text())
    assert second['status']=='complete' and second['elapsed_incomplete'] is True
    assert (tmp_path/'attempts/000_files/trajectory.jsonl').read_text()=='running original candidates\n'
    assert (tmp_path/'attempts/001_files/trajectory.jsonl').read_text()=='complete original candidates\n'


def test_user_scope_prevents_accidental_expansion_on_restart(tmp_path):
    from regulatory_design.launch import execution_limit
    assert execution_limit(tmp_path,'all')=='all'
    scope=tmp_path/'execution_scope.json'
    scope.write_text(json.dumps({'through':'pilot'}))
    assert execution_limit(tmp_path,'all')=='pilot'
    assert execution_limit(tmp_path,'calibration')=='calibration'
    scope.write_text(json.dumps({'through':'invalid'}))
    with pytest.raises(ValueError,match='执行范围无效'):
        execution_limit(tmp_path,'all')


def test_paper_materialization_and_raw_tamper_detection(tmp_path,monkeypatch):
    from regulatory_design import paper
    from regulatory_design.report import summarize
    root=tmp_path/'results';root.mkdir()
    stages={}
    for stage,length in [('formal',6144),('long_validation',19968)]:
        pair=[]
        for backend,elapsed,batch in [('native',120.,8),('evspark',60.,4)]:
            source=root/f'{stage}_{backend}.json'
            row={'source':str(source),'total_s':elapsed,'sequence':'ACGT'*(length//4),
                 'qualified':True,'checker':{'checker_auc':.95},'guide':{'ensemble_auc':.96},
                 'length':length,'batch':batch,'peak_generation_bytes':2**30,'peak_scorer_bytes':2**31}
            source.write_text(json.dumps(row));pair.append(row)
        pairs=[tuple(pair)];summary=summarize(pairs)
        (root/f'{stage}_summary.json').write_text(json.dumps(summary))
        stages[stage]={'pairs':pairs,'summary':summary,
                       'selection':{'selection':{'native':{'batch':8},'evspark':{'batch':4}}}}
    # 合成数据仅写 pytest 临时目录，用于验证入稿路径与防篡改，绝不写真实结果目录。
    monkeypatch.setattr(paper,'validated_stages',lambda _:stages)
    main=tmp_path/'paper/latex_v3/main.tex';main.parent.mkdir(parents=True)
    main.write_text('unchanged introduction\n'+paper.OLD_SCOPE+'\n'+paper.OLD_REMAINING+
                    '\n\\section{Related work}\nunchanged related work\n\\begin{thebibliography}{10}\n\\end{thebibliography}\n')
    (root/'formal_main_figure.pdf').write_bytes(b'test-only figure placeholder')
    paper.materialize(root,tmp_path)
    text=main.read_text()
    assert '2.00$\\times$' in text
    assert '6,144 bp & native & 8 & 2.00 & 1.00 & 2.00' in text
    assert 'unchanged introduction' in text and 'unchanged related work' in text
    assert paper.NEW_SCOPE in text and paper.OLD_SCOPE not in text
    assert paper.audit(root,tmp_path)['n_source_designs']==4
    paper.materialize(root,tmp_path)
    assert main.read_text()==text
    assert main.read_text().count('\\bibitem{enformer}')==1
    source=root/'formal_native.json';source.write_text('{}')
    with pytest.raises(ValueError,match='原始结果已改变'):
        paper.audit(root,tmp_path)


def _toy_candidate_generator(monkeypatch, batch):
    """三符号、有完整历史依赖的 CPU target，运行生产候选循环。"""
    torch = pytest.importorskip('torch')
    from contextlib import nullcontext
    from types import SimpleNamespace
    if not torch.cuda.is_available():
        pytest.skip('小模型在 CPU 计算，但 Vortex 导入需要可见 CUDA 驱动')
    pytest.importorskip('vortex')
    from regulatory_design import generation as g
    transitions = np.array([[.1,.3,.6],[.55,.35,.1],[.25,.5,.25]])
    gen = object.__new__(g.CandidateGenerator)
    gen.device,gen.batch,gen.gamma,gen.max_length = torch.device('cpu'),batch,3,1000
    gen.native_only = False
    gen.capture = SimpleNamespace(begin=lambda:None)
    gen.drafter = SimpleNamespace(ctx_window=4)
    gen.meter = SimpleNamespace(stage=lambda _:nullcontext())

    def restore(parent, width):
        gen.ip = {name:SimpleNamespace(seqlen_offset=parent.length) for name in ('mha','hcl','hcm','hcs')}
        gen.ip['mha'].lengths_per_sample = torch.full((width,),parent.length,dtype=torch.int32)
        gen.histories = [list(parent.history) for _ in range(width)]

    def model(tokens, inference_params_dict):
        logits, hidden = [], []
        for row, chunk in enumerate(tokens.tolist()):
            values, features = [], []
            for token in chunk:
                gen.histories[row].append(token)
                state = sum(gen.histories[row]) % 3
                values.append(np.log(transitions[state]))
                features.append([token,state])
            logits.append(values); hidden.append(features)
        gen.hidden = torch.tensor(hidden,dtype=torch.float32)
        return torch.tensor(np.array(logits),dtype=torch.float32),None

    def extract(row, length, anchor, context, parent):
        assert len(gen.histories[row]) == length
        return SimpleNamespace(length=length,anchor=anchor,context=context.clone(),
                               history=tuple(gen.histories[row]))

    def block(model_arg,tokens,ip,retain):
        before = [len(row) for row in gen.histories]
        logits,_ = model_arg(tokens,ip)
        return logits,before

    def accept(ip,before,accepted):
        for row,k in enumerate(accepted):
            del gen.histories[row][before[row]+int(k)+1:]

    def draft(anchors,contexts,rngs,greedy):
        # 有意使用低接受率 q，触发行间接受长度不同及提前结束。
        q = np.broadcast_to([.8,.1,.1],(len(rngs),gen.gamma,3)).copy()
        tokens=np.array([[g.sample_categorical(prob,rng) for prob in row] for row,rng in zip(q,rngs)])
        return tokens,q

    gen._restore,gen._extract,gen.model,gen._draft = restore,extract,model,draft
    gen._captured = lambda:gen.hidden
    monkeypatch.setattr(g,'block_forward',block)
    monkeypatch.setattr(g,'slice_states_to_accept',accept)
    monkeypatch.setattr(g,'_flash_kvcache_enabled',lambda _:True)
    parent = SimpleNamespace(length=3,anchor=1,context=torch.ones(4,2),history=(1,2,0))
    return gen,parent,transitions


def test_candidate_loop_batch_invariance_and_completed_state_isolation(monkeypatch):
    import copy
    for backend in ('native','evspark'):
        gen,parent,_ = _toy_candidate_generator(monkeypatch,4)
        seeds = [32,57,190,829,906,7102,8301]
        grouped = gen.generate_candidates(parent,seeds,length=11,backend=backend)
        gen.batch = 1
        scalar = gen.generate_candidates(parent,seeds,length=11,backend=backend)
        for a,b in zip(grouped,scalar):
            assert a.token_ids == b.token_ids
            assert a.state.history == b.state.history == parent.history+(parent.anchor,)+tuple(a.token_ids[:-1])
            assert a.state.length == parent.length+11
            assert a.rng_state == b.rng_state
            assert a.trace == b.trace
            assert np.array_equal(a.state.context,b.state.context)
        # 后续使用同一 arena 不得改写之前已经返回的候选状态或 RNG。
        before=copy.deepcopy([(r.state.history,r.rng_state) for r in grouped])
        gen.generate_candidates(parent,[18,29],length=17,backend=backend)
        assert before == [(r.state.history,r.rng_state) for r in grouped]
        assert parent.history == (1,2,0)
        if backend=='evspark':
            assert len({len(r.trace) for r in grouped})>1


def test_candidate_loop_small_vocabulary_joint_distribution(monkeypatch):
    # 枚举两步完整序列概率，并独立检查相邻两候选的首符号联合分布。
    n=3000
    for backend in ('native','evspark'):
        gen,parent,p = _toy_candidate_generator(monkeypatch,4)
        rows=gen.generate_candidates(parent,list(range(10000,10000+n)),length=2,backend=backend)
        sequences=np.array([r.token_ids for r in rows])
        initial=(sum(parent.history)+parent.anchor)%3
        exact=np.array([[p[initial,a]*p[(initial+a)%3,b] for b in range(3)] for a in range(3)])
        observed=np.bincount(sequences[:,0]*3+sequences[:,1],minlength=9).reshape(3,3)/n
        paired=sequences[:,0].reshape(-1,2)
        joint=np.bincount(paired[:,0]*3+paired[:,1],minlength=9).reshape(3,3)/(n/2)
        assert np.max(np.abs(observed-exact))<.035
        assert np.max(np.abs(joint-np.outer(p[initial],p[initial])))<.045
