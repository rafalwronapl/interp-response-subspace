"""Prepare real E031 execution dependencies; RAM gate can close before pretrained work."""
import shutil
from pathlib import Path
from research_checks.gate import read,create,digest,environment_identity
from research_checks.persistence import immutable
from research_checks.reachability import SEMANTICS,analyze
from e012.common import save,sha,rows
from e024.common import payload
from e024.prepare import source_closure
from .common import ROOT,OUT,PREP,CONFIG,guard,donor,memory_requirement


def resources():
    import torch
    from .subject import Subject
    from .loading import load
    from e017.common import memory
    m=memory();ok=True;error=None
    try:guard(start=True)
    except RuntimeError as exc:ok=False;error=str(exc)
    record=dict(ok=ok,measurement_after_imports=True,memory=m,free_disk_GiB=shutil.disk_usage(ROOT).free/2**30,
        requirement=memory_requirement(),user_authorized_departure_from_10GiB=True,error=error,model_loaded=False,pretrained_calls=0)
    save(PREP/'resource_gate.json',record);print(record)


def build():
    if (OUT/'ledger.jsonl').exists():raise RuntimeError('No rebuild after pretrained work')
    OUT.mkdir(parents=True,exist_ok=True);cfg=read(CONFIG);data=read(PREP/'dataset.json')
    # Actual historical baseline inputs and exposure sources used by design preparation.
    files=read(ROOT/'analysis/E030_preparation/design_inputs_v2.json')['files'].copy()
    def bind(path):
        p=Path(path);files[p.relative_to(ROOT).as_posix()]=sha(p);return p
    for p,h in files.items():
        if sha(ROOT/p)!=h:raise ValueError('Historical input changed: '+p)
    from transformers import AutoTokenizer
    folder=ROOT/cfg['model']['local_path'];tok=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    for dd in (data['pilot_documents'],data['documents']):
        for d in dd:
            if tok.encode(d['prompt'],add_special_tokens=False)!=d['ids']:raise ValueError('Tokenizer changed')
            other=donor(d,dd)
            if len(d['ids'])!=len(other['ids']) or d['positions']!=other['positions']:raise ValueError('Donor alignment')
            if any(a!=b for i,(a,b) in enumerate(zip(d['ids'],other['ids'])) if i not in d['positions']['subject']):raise ValueError('Other token changed')
            for ans,target in [(' is',374),(' are',525)]:
                if tok.encode(d['prompt']+ans,add_special_tokens=False)!=d['ids']+[target]:raise ValueError('Readout changed')
    # Materialize only the historical rows needed for exact pilot reproduction.
    bind(ROOT/'runs/E014/measurements.jsonl')
    records={r['label']:r for r in rows(ROOT/'runs/E014/measurements.jsonl')};history={}
    for d in data['pilot_documents']:
        history[d['id']]={}
        for key,label in [('natural','B'),('b','H111'),('c','H111_W0')]:
            rec=records[d['id']+'__'+label]
            p=bind(ROOT/'runs/E014'/rec['cache'])
            if sha(p)!=rec['cache_sha256']:raise ValueError('Historical raw changed')
            target=PREP/'historical'/f'{d["id"]}_{key}.json';save(target,rec);bind(target)
            history[d['id']][key]=target.relative_to(ROOT).as_posix()
    save(PREP/'historical_map.json',history)
    graphs=[]
    for stage,key in [('pilot','pilot_documents'),('main','documents')]:
        for d in data[key]:
            last=len(d['ids'])-1
            # This graph states possible residual-skip reachability; exact full MLP20
            # clamp and differentiable unit semantics have separate proofs/tests.
            spec=dict(semantics=SEMANTICS,layers=28,heads_per_layer=16,sequence_length=len(d['ids']),
                dynamic_heads=[[l,h] for l in range(28) for h in range(16)],dynamic_mlps=[l for l in range(28) if l!=20],
                source=dict(layer=19,phase='mlp_output',positions=[last]),
                targets=[dict(id='W21_input',layer=21,phase='mlp_input',positions=[last]),dict(id='readout',layer=27,phase='residual_out',positions=[last])])
            if not analyze(spec)['ok']:raise ValueError('Required downstream checkpoint unreachable')
            path=PREP/'circuits'/f'{stage}_{d["id"]}.json';save(path,spec);graphs.append(dict(stage=stage,id=path.stem,spec=path.relative_to(ROOT).as_posix()))
        schedule=read(PREP/f'schedule_{stage}.json');save(PREP/f'schedule_execution_{stage}.json',dict(operations=schedule['operations']))
    save(PREP/'graphs.json',graphs)
    save(PREP/'structure_proof.json',dict(ok=True,scope='Native scalar product coordinate U19 final token; residual skip reaches W21 and readout despite full MLP20 output clamp. Earlier positions remain identical by causality. W clamp removes derivative through that coordinate while all others and attention remain dynamic. Topology graph is a necessary reachability embedding, not a scalar-value or path-specific adapter.',gradients='Input t is leaf; u=u0+t*delta only once; no prefix graph or weight grad. clone/index assignment preserves derivative; MLP20/W constants intentionally block only their respective branches.'))
    sources=source_closure(['e031.worker','e031.subject','e031.common','e031.report','e031.prepare','e031.tests'])
    code={p.relative_to(ROOT).as_posix():sha(p) for p in sources};files.update(code)
    model={p.relative_to(ROOT).as_posix():sha(p) for p in folder.iterdir() if p.is_file()};files.update(model)
    if model[cfg['model']['local_path']+'/model.safetensors']!=cfg['model']['weights_sha256']:raise ValueError('Actual weights mismatch')
    identity=dict(cfg['model'],environment_sha256=digest(environment_identity()),all_model_files_sha256=digest(model),measurement_config_sha256=sha(CONFIG),measurement_before_answer=True)
    save(PREP/'identity.json',dict(model_identity=identity,code_identity=digest(code),source_files=code))
    immutable(OUT/'authorization.json',dict(user_message='rob',scope='Implement and execute E031 only through separately reviewed stages and fixed resource/precision gates, local CPU/no paid API.'))
    # Array persistence stores only selected vectors/scalars, never a full vocabulary
    # logit vector, and one detached natural reference at a time.
    required=max(4*2**30,1392*(2048*3*4+40000)+2*2**30)
    save(PREP/'resources_execution.json',dict(ok=shutil.disk_usage(ROOT).free>=required,required_disk_bytes=required,free_disk_bytes=shutil.disk_usage(ROOT).free,
        planned_forward=1248,planned_backward=640,weights_resident_RAM_bytes=4*1720_000_000,API_USD=0,checkpoint='Only U/W scalars, MLP20 before/after and final hidden; compressed NPZ, no graphs saved'))
    for p in [CONFIG,PREP/'dataset.json',PREP/'frozen_baselines.json',PREP/'historical_map.json',PREP/'identity.json',PREP/'graphs.json',PREP/'structure_proof.json',
              PREP/'implementation_tests_v2.json',PREP/'resource_gate.json',PREP/'resources_execution.json',PREP/'schedule_pilot.json',PREP/'schedule_main.json',OUT/'authorization.json',
              ROOT/'plans/E031_MEMORY_AMENDMENT.md',
              ROOT/'plans/E031_PLAN.md',ROOT/'plans/E030_E031_HANDOFF.md',ROOT/'analysis/E030_E031_design_review/snapshot_v3.json',ROOT/'analysis/E030_E031_design_review/review_v3.json']:
        bind(p)
    save(PREP/'execution_input_manifest.json',dict(files=files))
    save(PREP/'preflight_execution.json',dict(ok=True,pretrained_calls=0,pilot_documents=32,main_documents=128,graphs=len(graphs),
        resource_gate_open=read(PREP/'resource_gate.json')['ok'],implementation_tested_without_pretrained_weights=True))
    print(dict(prepared=True,source_files=len(code),inputs=len(files),resource_gate_open=read(PREP/'resource_gate.json')['ok']))


def contract(stage):
    gates=[dict(id=k,category=k,evidence='analysis/E031_preparation/implementation_tests_v2.json',pointer='/'+v,op='eq',expected=True,reason='Actual random Qwen differentiation, scalar analytical example, cache and mixed-kind budget controls') for k,v in [('hooks','hooks_ok'),('data','data_ok'),('pilot','pilot_ready'),('accounting','accounting_ok'),('restart','restart_ok')]]
    gates += [dict(id='load_resources',category='accounting',evidence='analysis/E031_preparation/resource_gate.json',pointer='/ok',op='eq',expected=True,reason='User-authorized dynamic streaming peak plus512MiB reserve, checked after imports and before load')]
    manifest='analysis/E031_preparation/execution_input_manifest.json'
    if stage=='main':
        if not payload(OUT/'pilot_results.json')['ready'] or not read(OUT/'audit_pilot.json')['ok']:raise RuntimeError('Pilot gate closed')
        immutable(PREP/'main_prior_ledger.json',list(rows(OUT/'ledger.jsonl')))
        files=read(PREP/'execution_input_manifest.json')['files'].copy()
        include=[p for folder in ('raw','index','arrays') for p in (OUT/folder).glob('*') if p.is_file()]
        include += [PREP/'main_prior_ledger.json',OUT/'pilot_results.json',OUT/'pilot_forecasts.json',OUT/'pilot_controls.json',OUT/'audit_pilot.json',PREP/'pilot_snapshot_v1.json',PREP/'pilot_review_v1.json']
        files.update({p.relative_to(ROOT).as_posix():sha(p) for p in include});save(PREP/'main_input_manifest.json',dict(files=files));manifest='analysis/E031_preparation/main_input_manifest.json'
        gates += [dict(id='pilot_signal',category='pilot',evidence='runs/E031/pilot_results.json',pointer='/payload/ready',op='eq',expected=True,reason='Fixed signal plus precision gate'),dict(id='pilot_audit',category='pilot',evidence='runs/E031/audit_pilot.json',pointer='/ok',op='eq',expected=True,reason='Raw-first forecast and numerical reconstruction')]
    graphs=[{k:v for k,v in g.items() if k!='stage'} for g in read(PREP/'graphs.json') if g['stage']==stage]
    save(PREP/f'{stage}.contract.json',dict(schema_version=1,id='E031_'+stage,kind='experiment',primary_agent='/root',purpose='Fixed natural-only Taylor2 transfer '+stage,
        tracked_files=['plans/E031_PLAN.md','plans/E031_config.json','plans/E031_MEMORY_AMENDMENT.md'],tracked_trees=['e031'],input_manifests=[manifest],gates=gates,circuits=graphs,
        structural_scope='Topology embedding for residual-skip reachability with MLP20 frozen; scalar native-unit values/gradients and position-local patches separately proved and random-Qwen tested.',
        entrypoint=dict(module='e031.worker',args=['--stage',stage]),schedule=f'analysis/E031_preparation/schedule_execution_{stage}.json',budget=read(CONFIG)['budget']['stage_caps'][stage]))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--resources',action='store_true');p.add_argument('--contract',choices=['pilot','main']);a=p.parse_args()
    if a.resources:resources()
    elif a.contract:contract(a.contract)
    else:build()
