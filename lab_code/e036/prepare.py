"""Freeze inputs before inference; generate separately reviewed stage contracts."""
import json
import re
import sys
from pathlib import Path
from research_checks.gate import read,sha,digest,environment_identity,create
from .common import ROOT,PREP,OUT,CAPS,TIMES,OFFLINE,guard
from .data import build,operations

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def history():
    files=[];prompts=set()
    def visit(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if k in ('prompt','text') and isinstance(v,str):prompts.add(v)
                else:visit(v)
        elif isinstance(obj,list):
            for v in obj:visit(v)
    for folder in ('runs','data','analysis'):
        for p in sorted((ROOT/folder).rglob('*dataset*.json')):
            if 'E036' in str(p) or p.stat().st_size>20*2**20:continue
            obj=read(p);visit(obj);files.append(p)
    return files,prompts

def base():
    import time
    preparation_start=time.monotonic()
    if any(OUT.rglob('ledger.jsonl')):raise RuntimeError('Inputs already exposed')
    from e024.prepare import source_closure
    PREP.mkdir(parents=True,exist_ok=True)
    data=build();hist,prompts=history()
    overlap=[d['id'] for d in data['documents'] if d['split']!='pilot' and d['prompt'] in prompts]
    if overlap:raise ValueError('Exact historical prompt overlap '+str(overlap))
    vocabulary=set(re.findall(r'[a-z]+',' '.join(prompts).lower()))
    exposure={w:dict(singular_in_history=w in vocabulary,plural_in_history=w+'s' in vocabulary) for w in data['lemmas']}
    write(PREP/'dataset.json',data)
    write(PREP/'data_checks.json',dict(ok=True,counts={s:sum(d['split']==s for d in data['documents']) for s in ('pilot','fit','select','test')},
          exact_prompt_overlap=overlap,lexical_exposure=exposure,history_files=[str(p.relative_to(ROOT)) for p in hist],
          scope='Fresh prompts and disjoint study lemmas across splits; historical words explicitly flagged, not claimed unseen in pretraining.'))
    for stage in CAPS:
        cats=operations(data['documents'],stage)
        counts={k:sum(c==k for c in cats.values()) for k in CAPS[stage]}
        if any(counts[k]>CAPS[stage][k] for k in counts):raise ValueError('Operation cap arithmetic')
        write(PREP/(stage+'_operations.json'),dict(categories=cats,counts=counts))
        write(PREP/(stage+'_schedule.json'),dict(operations=[dict(id=k,kind='forward') for k in cats]))
    paths=source_closure(['e036.prepare','e036.worker','e036.tests','e036.report'])
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}
    model=read(ROOT/'plans/E035_config.json')['model']
    weights=sha(ROOT/'model/model.safetensors')
    if weights!=model['weights_sha256']:raise ValueError('Wrong model')
    upstream=read(ROOT/'runs/E017/upstream.json')
    if upstream['sha']!='5301dca1909eccac1ecedd3d58da60097056db8f':raise ValueError('Wrong atlas')
    assets=[]
    for layer in (20,21):
        for role,folder in [('enc','e017'),('dec','e033')]:
            name=f'W_{role}_{layer}.safetensors';p=ROOT/f'clt_cache/{folder}'/name
            spec=next(s for s in upstream['siblings'] if s['rfilename']==name)
            if sha(p)!=spec['lfs']['sha256']:raise ValueError('Wrong atlas asset')
            assets.append(p)
    model_files=[p for p in (ROOT/'model').iterdir() if p.is_file()]
    manifest_files=hist+paths+model_files+assets+[PREP/'IMPLEMENTACJA.txt',ROOT/'model_download.json',ROOT/'plans/E035_config.json',ROOT/'runs/E017/upstream.json',ROOT/'plans/E036_PLAN.txt',ROOT/'plans/E036_hypothesis.json']
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(set(manifest_files))}
    write(PREP/'identity.json',dict(model_identity=dict(model,atlas_revision=upstream['sha'],assets={str(p.relative_to(ROOT)):sha(p) for p in assets},model_files={str(p.relative_to(ROOT)):sha(p) for p in model_files}),
          protocol_identity=dict(protocol_sha256=sha(ROOT/'plans/E036_PLAN.txt'),dataset_sha256=sha(PREP/'dataset.json'),prefix=151643,readout=[374,525]),
          code_identity=digest(sources),source_files=sources,environment_sha256=digest(environment_identity())))
    write(PREP/'base_inputs.json',dict(files=hashes))
    write(PREP/'resources.json',dict(ok=True,start=guard(start=True),estimated_output_mib=160,
          model_seconds=TIMES,offline_seconds=OFFLINE,preparation_hashing_tokenizer_seconds=time.monotonic()-preparation_start,
          source='Analytic shape bound; actual pilot must confirm.'))
    print(json.dumps({'prepared':True,'bound_files':len(hashes),'stages':{s:read(PREP/(s+'_operations.json'))['counts'] for s in CAPS}}))

def contract(stage,version='v1'):
    from e024.prepare import source_closure
    identity=read(PREP/'identity.json')
    current={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in source_closure(['e036.prepare','e036.worker','e036.tests','e036.report'])}
    if digest(current)!=identity['code_identity']:raise ValueError('Code changed since preparation')
    files=dict(read(PREP/'base_inputs.json')['files'])
    repair=identity.get('pilot_repair') if stage=='pilot' else None
    opfile=ROOT/repair['operations_file'] if repair and repair.get('operations_file') else PREP/(stage+'_operations.json')
    schedulefile=repair['schedule_file'] if repair and repair.get('schedule_file') else 'analysis/E036_preparation/'+stage+'_schedule.json'
    additional=[PREP/'identity.json',PREP/'dataset.json',PREP/'data_checks.json',PREP/'resources.json',opfile]
    if stage!='pilot':
        for previous in (['pilot'] if stage=='development' else ['pilot','development']):
            additional.extend(p for p in (OUT/previous).rglob('*') if p.is_file() and p.suffix not in ('.log','.tmp'))
            if previous=='pilot' and identity.get('pilot_repair'):
                additional.extend(p for p in (ROOT/identity['pilot_repair']['new_cache']).rglob('*') if p.is_file() and p.suffix not in ('.log','.tmp'))
    for p in additional:files[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
    name=f'{stage}_inputs_{version}.json';create(PREP/name,dict(files=files))
    gates=[]
    for category,key in [('hooks','hooks_ok'),('data','data_ok'),('pilot','pilot_ready'),('accounting','accounting_ok'),('restart','restart_ok')]:
        gates.append(dict(id=category,category=category,evidence='analysis/E036_preparation/tests.json',pointer='/'+key,op='eq',expected=True,reason='Concrete random-model, leakage, prediction, accounting and hostile persistence checks.'))
    for path,key,label,category in [(PREP/'data_checks.json','ok','literal_data','data'),(PREP/'resources.json','ok','resources','accounting')]:
        gates.append(dict(id=label,category=category,evidence=str(path.relative_to(ROOT)).replace('\\','/'),pointer='/'+key,op='eq',expected=True,reason='Actual pinned tokenizer, historical overlap and resource evidence.'))
    if stage!='pilot':
        gates.append(dict(id='trained_pilot',category='pilot',evidence='runs/E036/pilot/checks.json',pointer='/ready',op='eq',expected=True,reason='Actual trained-model technical pilot.'))
    if stage=='test':
        gates.append(dict(id='select_signal',category='pilot',evidence='runs/E036/development/checks.json',pointer='/select_signal',op='eq',expected=True,reason='Frozen SELECT signal gate; not a candidate-success gate.'))
    obj=dict(schema_version=1,id='E036_'+stage+'_'+version,kind='experiment',primary_agent='/root',purpose='E036 prospective local CLT differential error prediction: '+stage,
        tracked_files=['plans/E036_PLAN.txt','plans/E036_hypothesis.json'],tracked_trees=['e036'],input_manifests=['analysis/E036_preparation/'+name],gates=gates,circuits=[],
        structural_scope='Original dynamic Qwen prefix. X19 product last -> MLP20 -> x21; restore M20 is at product before down_proj. Captures before MLP21. Local target MLP21 acts per token. No reduced circuit or frozen attention. Earlier positions invariant; sources <=19 encoder inputs precede X patch.',
        entrypoint=dict(module='e036.worker',args=[stage]),schedule=schedulefile,budget=dict(forward=sum(CAPS[stage].values()),backward=0))
    create(PREP/f'{stage}_{version}.contract.json',obj)
    print(json.dumps({'contract':str(PREP/f'{stage}_{version}.contract.json')}))

def refresh_code(version='v1'):
    """Before inference only: retain the old identities and bind edited sources anew."""
    from e024.prepare import source_closure
    if any(OUT.rglob('ledger.jsonl')):raise RuntimeError('Cannot change identity after model execution')
    identity=read(PREP/'identity.json');base=read(PREP/'base_inputs.json')
    create(PREP/f'identity_before_code_update_{version}.json',identity)
    create(PREP/f'inputs_before_code_update_{version}.json',base)
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in source_closure(['e036.prepare','e036.worker','e036.tests','e036.report'])}
    identity['source_files']=sources;identity['code_identity']=digest(sources)
    base['files'].update(sources)
    write(PREP/'identity.json',identity);write(PREP/'base_inputs.json',base)

def repair_pilot():
    """Explicit technical amendment after failed pilot, before any main data."""
    import ast
    from .common import payload,events
    from e024.prepare import source_closure
    if (OUT/'development').exists() or (OUT/'test').exists():raise RuntimeError('Main-data exposure prohibits pilot repair')
    if (OUT/'pilot/complete.json').exists():raise RuntimeError('Pilot already completed')
    archive=PREP/'before_v3';old=read(archive/'identity.json');identity=read(PREP/'identity.json')
    if old['model_identity']!=identity['model_identity'] or old['protocol_identity']!=identity['protocol_identity']:raise ValueError('Model/protocol changed')
    def fingerprint(path,cls,method):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
        function=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==method)
        return digest(ast.dump(function,include_attributes=False))
    footprints={}
    for cls,method in [('Subject','__init__'),('Subject','forward'),('Subject','jvp'),('Subject','close'),('Atlas','__init__'),('Atlas','evaluate')]:
        a=fingerprint(archive/'e036/subject.py',cls,method);b=fingerprint(ROOT/'e036/subject.py',cls,method)
        if a!=b:raise ValueError('Cannot adopt changed computation '+cls+'.'+method)
        footprints[cls+'.'+method]=a
    labels=[]
    for p in (OUT/'pilot/index').glob('*.json'):
        ix=payload(p)
        if ix['request']['code_identity']!=old['code_identity']:raise ValueError('Mixed pilot provenance')
        if ix['request']['intervention']['category']!='target':labels.append(p.stem)
    records=events(OUT/'pilot/ledger.jsonl')
    counts={k:sum(e['event']=='reserve' and e['category']==k for e in records) for k in CAPS['pilot']}
    seconds=sum(e['seconds'] for e in records if e['event']=='finish')
    if counts!={'network':42,'target':32,'jvp':32,'clt':40}:raise ValueError('Unexpected pilot carry counts')
    if len(labels)!=114:raise ValueError('Unexpected adoption inventory')
    proof=dict(reason='Identical-input local/full parity; native-shape FP32 GEMM. No tolerance or scientific criterion change.',
        unchanged_function_AST_sha256=footprints,adoptable_labels=sorted(labels),counts_already_spent=counts,seconds_already_spent=seconds,
        planned_combined_counts={'network':88,'target':120,'jvp':40,'clt':40},old_sources_sha256=old['code_identity'],
        old_ledger_sha256=sha(OUT/'pilot/ledger.jsonl'),no_main_exposure=True)
    write(PREP/'pilot_repair_v3.json',proof)
    identity['pilot_repair']=dict(old_cache='runs/E036/pilot',new_cache='runs/E036/pilot_v3',carry_ledger='runs/E036/pilot/ledger.jsonl',
        old_code_identity=old['code_identity'],adoptable_labels=sorted(labels),equivalence_sha256=sha(PREP/'pilot_repair_v3.json'))
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in source_closure(['e036.prepare','e036.worker','e036.tests','e036.report'])}
    identity['source_files']=sources;identity['code_identity']=digest(sources)
    base=read(PREP/'base_inputs.json');base['files'].update(sources)
    extra=[p for p in (OUT/'pilot').rglob('*') if p.is_file()]+[p for p in archive.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    extra += [PREP/'pilot_repair_v3.json',PREP/'pilot_snapshot_v2.json',ROOT/'analysis/E036_execution_review/pilot_review_v2.json']
    for p in extra:base['files'][str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
    write(PREP/'identity.json',identity);write(PREP/'base_inputs.json',base)
    print(json.dumps(proof))

def repair_auxiliary_gate():
    """Keep the same-input criterion; replace a mismatched-operator auxiliary check."""
    import ast
    from .common import payload,events
    from .data import operations
    from e024.prepare import source_closure
    if (OUT/'development').exists() or (OUT/'test').exists() or (OUT/'pilot/complete.json').exists():raise RuntimeError('Repair scope closed')
    archive=PREP/'before_v4';old=read(archive/'identity.json');identity=read(PREP/'identity.json')
    if old['model_identity']!=identity['model_identity'] or old['protocol_identity']!=identity['protocol_identity']:raise ValueError('Model/protocol changed')
    def fp(path,cls,method):
        tree=ast.parse(path.read_text(encoding='utf-8'));body=tree.body
        if cls:body=next(n for n in body if isinstance(n,ast.ClassDef) and n.name==cls).body
        node=next(n for n in body if isinstance(n,ast.FunctionDef) and n.name==method)
        return digest(ast.dump(node,include_attributes=False))
    footprints={}
    methods=[('subject.py','Subject',m) for m in ('__init__','forward','jvp','local','close')]+[('subject.py','Atlas',m) for m in ('__init__','evaluate')]+[('worker.py',None,m) for m in ('natural','collect','target')]
    for name,cls,method in methods:
        a=fp(archive/'e036'/name,cls,method);b=fp(ROOT/'e036'/name,cls,method)
        if a!=b:raise ValueError('Changed scientific computation '+str((cls,method)))
        footprints[name+':'+str(cls)+'.'+method]=a
    labels=[]
    for p in (OUT/'pilot_v3/index').glob('*.json'):
        ix=payload(p)
        if ix['request']['code_identity']!=old['code_identity']:raise ValueError('Mixed v3 source identity')
        if ix['request']['intervention']['category']=='target' and not ix['record']['path'].replace('\\','/').startswith('runs/E036/pilot_v3/arrays/'):
            raise ValueError('V2 targets cannot be adopted')
        labels.append(p.stem)
    if len(labels)!=160:raise ValueError('Unexpected v3 inventory')
    ledgers=['runs/E036/pilot/ledger.jsonl','runs/E036/pilot_v3/ledger.jsonl']
    prior=[e for path in ledgers for e in events(ROOT/path)]
    counts={k:sum(e['event']=='reserve' and e['category']==k for e in prior) for k in CAPS['pilot']}
    if counts!={'network':48,'target':71,'jvp':33,'clt':40}:raise ValueError('Carry count mismatch')
    cats=operations(read(PREP/'dataset.json')['documents'],'pilot')
    write(PREP/'pilot_operations_v4.json',dict(categories=cats,counts={k:sum(v==k for v in cats.values()) for k in CAPS['pilot']}))
    write(PREP/'pilot_schedule_v4.json',dict(operations=[dict(id=k,kind='forward') for k in cats]))
    proof=dict(reason='Required same-input local/full gate retained; native PRE full control replaces invalid alternative-FP32-operator gate. Alternative arithmetic drift is descriptive.',
        unchanged_scientific_AST_sha256=footprints,adoptable_labels=sorted(labels),counts_already_spent=counts,
        seconds_already_spent=sum(e['seconds'] for e in prior if e['event']=='finish'),carry_ledgers_sha256={p:sha(ROOT/p) for p in ledgers},
        planned_combined_counts={'network':96,'target':120,'jvp':40,'clt':40},old_sources_sha256=old['code_identity'],
        unchanged_tolerances=dict(rtol=1e-5,atol=1e-6),no_main_exposure=True,v2_targets_excluded=True)
    write(PREP/'pilot_repair_v4.json',proof)
    identity['pilot_repair']=dict(old_cache='runs/E036/pilot_v3',new_cache='runs/E036/pilot_v4',carry_ledgers=ledgers,
        old_code_identity=old['code_identity'],adoptable_labels=sorted(labels),adopt_categories=['network','target','jvp','clt'],
        equivalence_sha256=sha(PREP/'pilot_repair_v4.json'),operations_file='analysis/E036_preparation/pilot_operations_v4.json',
        schedule_file='analysis/E036_preparation/pilot_schedule_v4.json')
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in source_closure(['e036.prepare','e036.worker','e036.tests','e036.report'])}
    identity['source_files']=sources;identity['code_identity']=digest(sources)
    base=read(PREP/'base_inputs.json');base['files'].update(sources)
    extra=[p for p in (OUT/'pilot_v3').rglob('*') if p.is_file()]+[p for p in archive.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    extra += [PREP/'pilot_repair_v4.json',PREP/'pilot_operations_v4.json',PREP/'pilot_schedule_v4.json',PREP/'pilot_snapshot_v3.json',ROOT/'analysis/E036_execution_review/pilot_review_v3.json']
    for p in extra:base['files'][str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
    write(PREP/'identity.json',identity);write(PREP/'base_inputs.json',base)
    print(json.dumps({'repair':'v4','carry_counts':counts,'adoptable':len(labels),'planned_combined_counts':proof['planned_combined_counts']}))

if __name__=='__main__':
    if len(sys.argv)==1:base()
    elif sys.argv[1]=='refresh':refresh_code(*sys.argv[2:])
    elif sys.argv[1]=='repair':repair_pilot()
    elif sys.argv[1]=='repair4':repair_auxiliary_gate()
    else:contract(*sys.argv[1:])
