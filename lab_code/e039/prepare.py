"""Execution provenance and separate stage contracts; no ungated model calls."""
import json
import shutil
import sys
from .common import ROOT,PREP,DATA,OUT,CAPS,TIMES,guard,read,sha,digest,create,environment_identity,payload,events

def operations(docs,stage):
    prepared=read(ROOT/'analysis/E039_preparation/operations.json')['stages'][stage]
    original=[d for d in read(DATA)['documents'] if d['split']==stage]
    if len(docs)!=len(original):raise ValueError('Unexpected document count')
    mapping={a['id']:b['id'] for a,b in zip(original,docs)}
    ops=[dict(o,document=mapping[o['document']]) for o in prepared['operations']]
    cats={o['id']:o['category'] for o in ops}
    if len(cats)!=len(ops):raise ValueError('Duplicate operation IDs')
    counts={k:sum(v==k for v in cats.values()) for k in CAPS[stage]}
    if counts!=prepared['counts'] or any(counts[k]>CAPS[stage][k] for k in counts):raise ValueError('Accounting mismatch')
    return dict(operations=ops,categories=cats,counts=counts)

def sources():
    from e024.prepare import source_closure
    return {p.relative_to(ROOT).as_posix():sha(p) for p in source_closure(['e039.prepare','e039.worker','e039.tests','e039.report'])}

def base():
    if any(OUT.rglob('ledger.jsonl')):raise ValueError('Trained exposure already occurred')
    PREP.mkdir(parents=True,exist_ok=True);src=sources();files=dict(src)
    prior=read(ROOT/'analysis/E039_preparation/inputs.json')['files']
    for name,expected in prior.items():
        if sha(ROOT/name)!=expected:raise ValueError('Prepared source changed '+name)
    files.update(prior)
    tracked=['plans/E039_PLAN.txt','plans/E039_hypothesis.json','analysis/E039_preparation/dataset.json',
        'analysis/E039_preparation/prepare.py','analysis/E039_preparation/checks.json','analysis/E039_preparation/operations.json',
        'analysis/E039_preparation/predictor_inventory.json','analysis/E039_preparation/snapshot_v1.json',
        'analysis/E039_plan_review/review_v1.json',
        'plans/E035_config.json','runs/E017/upstream.json','model_download.json',
        'analysis/E039_execution/tests_v1.log']
    model=read(ROOT/'plans/E035_config.json')['model']
    if sha(ROOT/'model/model.safetensors')!=model['weights_sha256']:raise ValueError('Native identity')
    upstream=read(ROOT/'runs/E017/upstream.json')
    if upstream['sha']!='5301dca1909eccac1ecedd3d58da60097056db8f':raise ValueError('CLT revision')
    assets={}
    for layer in (20,21):
        for role,folder in [('enc','e017'),('dec','e033')]:
            name=f'W_{role}_{layer}.safetensors';path=f'clt_cache/{folder}/{name}'
            expected=next(x for x in upstream['siblings'] if x['rfilename']==name)['lfs']['sha256']
            if sha(ROOT/path)!=expected:raise ValueError('CLT asset identity')
            assets[path]=expected
    for p in (ROOT/'model').iterdir():
        if p.is_file():tracked.append(p.relative_to(ROOT).as_posix())
    for name in tracked:files[name]=sha(ROOT/name)
    files.update(assets)
    fitted=payload(ROOT/'runs/E036/development/predictors.json')
    if fitted['arrays']['sha256']!=sha(ROOT/'runs/E036/development/predictors.npz'):raise ValueError('Frozen parameters changed')
    identity=dict(model_identity=dict(native=model,atlas_revision=upstream['sha'],assets=assets,
        native_files={p:files[p] for p in tracked if p.startswith('model/')}),code_identity=digest(src),source_files=src,
        parameter_sha256=fitted['arrays']['sha256'],hidden_size=1024,row_counts=dict(pilot=48,main=576),
        protocol_identity=dict(plan_sha256=sha(ROOT/'plans/E039_PLAN.txt'),dataset_sha256=sha(DATA),
            parameters_sha256=fitted['arrays']['sha256'],readout='are525-is374'),environment_sha256=digest(environment_identity()))
    create(PREP/'identity.json',identity);create(PREP/'base_inputs.json',dict(files=files))
    for stage in CAPS:
        docs=[d for d in read(DATA)['documents'] if d['split']==stage]
        op=operations(docs,stage);create(PREP/(stage+'_operations.json'),op)
        create(PREP/(stage+'_schedule.json'),dict(operations=[dict(id=k,kind='forward') for k in op['categories']]))
    create(PREP/'resources.json',dict(ok=True,start=guard(True),estimated_main_output_gib=2.,time_caps=TIMES,
        scope='Pilot must confirm measured time and disk projection before MAIN.'))
    print(json.dumps(dict(prepared=True,files=len(files))))

def contract(stage,version):
    if sources()!=read(PREP/'identity.json')['source_files']:raise ValueError('Code changed after preparation')
    files=dict(read(PREP/'base_inputs.json')['files'])
    for name in ['identity.json','resources.json','tests.json',stage+'_operations.json',stage+'_schedule.json']:
        files[(PREP/name).relative_to(ROOT).as_posix()]=sha(PREP/name)
    resource_path=None
    if stage=='main':
        if not payload(OUT/'pilot/checks.json')['ready']:raise ValueError('Pilot gate closed')
        samples=[e['resources'] for e in events(OUT/'pilot/ledger.jsonl') if e['event']=='reserve' and e.get('resources')]
        if not samples:raise ValueError('No measured pilot resources')
        size=sum(p.stat().st_size for p in (OUT/'pilot').rglob('*') if p.is_file())
        projected=int(size*15);free=shutil.disk_usage(ROOT).free;current=guard(True)
        seconds=payload(OUT/'pilot/complete.json')['ledger']['seconds'];projected_seconds=seconds*15
        if projected+2**30>free:raise ValueError('Pilot disk projection exceeds capacity')
        if projected_seconds>TIMES['main']:raise ValueError('Pilot time projection exceeds MAIN cap')
        resource_path=PREP/('main_resources_'+version+'.json')
        create(resource_path,dict(ok=True,pilot_bytes=size,main_projected_bytes=projected,projection_factor=15,
            pilot_model_seconds=seconds,main_projected_seconds=projected_seconds,free_disk_bytes=free,current=current,
            pilot_peak_process_rss_gib=max(s['process_peak_rss_gib'] for s in samples),
            pilot_min_available_gib=min(s['available_gib'] for s in samples),pilot_min_commit_gib=min(s['commit_headroom_gib'] for s in samples)))
        files[resource_path.relative_to(ROOT).as_posix()]=sha(resource_path)
        for p in (OUT/'pilot').rglob('*'):
            if p.is_file():files[p.relative_to(ROOT).as_posix()]=sha(p)
        for p in (ROOT/'analysis/E039_execution_review').glob('pilot_*'):
            if p.is_file():files[p.relative_to(ROOT).as_posix()]=sha(p)
    name=f'{stage}_inputs_{version}.json';create(PREP/name,dict(files=files))
    gates=[]
    for category,key in [('hooks','hooks_ok'),('data','data_ok'),('pilot','pilot_ready'),('accounting','accounting_ok'),('restart','restart_ok')]:
        gates.append(dict(id=category,category=category,evidence='analysis/E039_execution/tests.json',pointer='/'+key,op='eq',expected=True,
            reason='Actual random Qwen bypass, stage flow, persistence and synthetic decision tests; trained pilot remains separate.'))
    gates.append(dict(id='resources',category='accounting',evidence='analysis/E039_execution/resources.json',pointer='/ok',op='eq',expected=True,reason='Actual local resource guard.'))
    if stage=='main':
        gates.append(dict(id='trained_pilot',category='pilot',evidence='runs/E039/pilot/checks.json',pointer='/payload/ready',op='eq',expected=True,reason='Frozen trained pilot signal and controls.'))
        gates.append(dict(id='resource_projection',category='accounting',evidence=resource_path.relative_to(ROOT).as_posix(),pointer='/ok',op='eq',expected=True,reason='Measured pilot time/disk and memory margins.'))
    create(PREP/f'{stage}_{version}.contract.json',dict(schema_version=1,id='E039_'+stage+'_'+version,kind='experiment',primary_agent='/root',
        purpose='Frozen MLP21 correction with dynamic native downstream: '+stage,tracked_files=['plans/E039_PLAN.txt','plans/E039_hypothesis.json'],
        tracked_trees=['e039'],input_manifests=['analysis/E039_execution/'+name],gates=gates,circuits=[],
        structural_scope='Native last-token X19/M20 prefix, MLP21 bypass, dynamic native downstream to is/are. No reduced circuit. Target-free counts, state equality and positive readout sensitivity required.',
        entrypoint=dict(module='e039.worker',args=[stage]),schedule='analysis/E039_execution/'+stage+'_schedule.json',budget=dict(forward=sum(CAPS[stage].values()),backward=0)))

if __name__=='__main__':
    if sys.argv[1]=='base':base()
    else:contract(sys.argv[1],sys.argv[2])
