"""Stage-local accounting and raw-first immutable, content-bound observations."""
import os
import time
import uuid
import shutil
from pathlib import Path
import numpy as np
from research_checks.gate import read, sha, digest, create
from research_checks.persistence import immutable, cached, request_key
from e017.common import memory

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT/'analysis/E036_preparation'
OUT = ROOT/'runs/E036'
CAPS = {'pilot': {'network':128,'target':128,'jvp':128,'clt':128},
        'development': {'network':704,'target':576,'jvp':576,'clt':704},
        'test': {'network':528,'target':432,'jvp':432,'clt':528}}
TIMES = {'pilot':300.,'development':1800.,'test':1500.}
OFFLINE = {'pilot':120.,'development':1080.,'test':600.}

def events(path):
    if not Path(path).exists(): return []
    return [__import__('json').loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]

def append(path, row):
    import json
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def payload(path):
    obj=read(path)
    if digest(obj['payload'])!=obj['payload_sha256']: raise ValueError('Corrupt receipt: '+str(path))
    return obj['payload']

def status(stage, **kw):
    import json
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'STATUS.tmp';p.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),**kw),allow_nan=False),encoding='utf-8')
    os.replace(p,OUT/'STATUS.json')

def guard(start=False):
    m=memory()
    if min(m['available_gib'],m['commit_headroom_gib']) < (4 if start else .5): raise RuntimeError('E036 RAM/commit guard')
    if shutil.disk_usage(ROOT).free < (2 if start else 1)*2**30: raise RuntimeError('E036 disk guard')
    if (OUT/'STOP').exists(): raise RuntimeError('E036 STOP requested')
    return m

def array_id(arrays):
    return {k:digest(dict(dtype=str(np.asarray(v).dtype),shape=list(np.asarray(v).shape),
                        bytes=__import__('hashlib').sha256(np.asarray(v).tobytes()).hexdigest())) for k,v in sorted(arrays.items())}

def save_arrays(folder, name, arrays):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/(name+'.npz');temp=folder/(name+'.tmp.npz')
    if path.exists(): raise ValueError('Refusing overwrite '+str(path))
    np.savez_compressed(temp,**arrays)
    os.link(temp,path);temp.unlink()
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),sha256=sha(path))

def load_arrays(record):
    p=Path(record['path']);p=p if p.is_absolute() else ROOT/p
    if sha(p)!=record['sha256']: raise ValueError('Array checksum mismatch')
    with np.load(p,allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
    if any(v.dtype.kind=='f' and not np.isfinite(v).all() for v in arrays.values()): raise ValueError('Nonfinite array')
    return arrays

class Budget:
    def __init__(self,stage,folder=None,allowed=None,resource_guard=guard,carry_ledger=None):
        self.stage=stage;self.folder=Path(folder) if folder else OUT/stage
        self.folder.mkdir(parents=True,exist_ok=True);self.log=self.folder/'ledger.jsonl'
        if allowed is not None:self.allowed=allowed
        else:
            repair=read(PREP/'identity.json').get('pilot_repair') if stage=='pilot' else None
            operation_path=ROOT/repair['operations_file'] if repair and repair.get('operations_file') else PREP/(stage+'_operations.json')
            self.allowed=read(operation_path)['categories']
        self.guard=resource_guard
        self.carry_ledgers=[Path(p) for p in carry_ledger] if isinstance(carry_ledger,list) else [Path(carry_ledger)] if carry_ledger else []
    def summary(self):
        ev=[e for path in self.carry_ledgers for e in events(path)]+events(self.log)
        r={x['id']:x for x in ev if x['event']=='reserve'};f={x['id']:x for x in ev if x['event']=='finish'}
        return dict(counts={k:sum(x['category']==k for x in r.values()) for k in CAPS[self.stage]},
                    seconds=sum(x['seconds'] for x in f.values()),failed=sum(not x['ok'] for x in f.values()),unfinished=len(set(r)-set(f)))
    def call(self,label,category,compute):
        if self.allowed.get(label)!=category: raise ValueError('Unscheduled operation '+label)
        s=self.summary()
        if s['unfinished']: raise RuntimeError('Unfinished reservation requires audited recovery')
        if s['counts'][category]>=CAPS[self.stage][category] or s['seconds']>=TIMES[self.stage]: raise RuntimeError('E036 budget cap')
        self.guard();ticket=uuid.uuid4().hex
        append(self.log,dict(event='reserve',id=ticket,label=label,category=category,utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
        start=time.monotonic();ok=False
        try:
            result=compute();ok=True;return result
        finally:
            seconds=time.monotonic()-start
            append(self.log,dict(event='finish',id=ticket,ok=ok,seconds=seconds))
            if s['seconds']+seconds>TIMES[self.stage]: raise RuntimeError('E036 time cap after operation')

class Measurements:
    def __init__(self,stage,budget,identity=None,folder=None):
        self.stage=stage;self.budget=budget;self.folder=Path(folder) if folder else OUT/stage
        self.identity=identity if identity is not None else read(PREP/'identity.json')
        fitted=OUT/'development/predictors.json'
        self.fit_sha=payload(fitted)['arrays']['sha256'] if stage=='test' and fitted.exists() else None
        self.forecast_verified=False
        self.adoption=self.identity.get('pilot_repair') if stage=='pilot' else None
    def run(self,label,category,doc,metadata,compute):
        request=dict(model_identity=self.identity['model_identity'],code_identity=self.identity['code_identity'],document=doc,
                     intervention=dict(label=label,category=category,metadata=metadata,stage=self.stage,
                         fit_sha256=self.fit_sha,protocol_identity=self.identity.get('protocol_identity')),readout=[374,525])
        key=request_key(request);index=self.folder/'index'/(label+'.json')
        if index.exists() and payload(index)['request']!=request: raise ValueError('Label provenance changed')
        def evaluate():
            if self.adoption and category in self.adoption.get('adopt_categories',['network','clt','jvp']) and label in self.adoption['adoptable_labels']:
                oldfolder=ROOT/self.adoption['old_cache']
                oldindex=oldfolder/'index'/(label+'.json');ix=payload(oldindex)
                expected=dict(request,code_identity=self.adoption['old_code_identity'])
                if ix['request']!=expected:raise ValueError('Adoption semantic request differs')
                oldraw=oldfolder/'raw'/(ix['key']+'.json')
                if payload(oldraw)!=dict(request=ix['request'],result=ix['record']):raise ValueError('Adoption raw/index mismatch')
                if request_key(ix['request'])!=ix['key']:raise ValueError('Adoption key mismatch')
                load_arrays(ix['record'])
                immutable(self.folder/'adoptions'/(label+'.json'),dict(old_index_sha256=sha(oldindex),old_raw_sha256=sha(oldraw),
                    old_code_identity=self.adoption['old_code_identity'],new_request_sha256=key,source_record=ix['record'],
                    equivalence_sha256=self.adoption['equivalence_sha256'],new_model_call=False))
                return ix['record']
            if self.stage=='test' and category=='target' and not self.forecast_verified:
                forecast=payload(OUT/'test/forecasts.json');predictions=load_arrays(forecast['arrays'])
                if forecast['rows']!=384: raise ValueError('Incomplete forecast freeze')
                if not predictions or any(v.shape!=(384,1024) for v in predictions.values()):raise ValueError('Forecast shape')
                self.forecast_verified=True
            arrays=self.budget.call(label,category,compute)
            if any(np.asarray(v).dtype.kind=='f' and not np.isfinite(v).all() for v in arrays.values()): raise ValueError('Nonfinite measurement')
            dest=self.folder/'arrays'/(key+'.npz')
            # An orphan after a crash is never accepted as an observation.
            if dest.exists(): raise ValueError('Orphan array requires audited recovery')
            return save_arrays(self.folder/'arrays',key,arrays)
        record=cached(self.folder/'raw',request,evaluate)
        immutable(index,dict(request=request,record=record,key=key))
        return load_arrays(record)
    def get(self,label):
        ix=payload(self.folder/'index'/(label+'.json'));raw=payload(self.folder/'raw'/(ix['key']+'.json'))
        if raw!=dict(request=ix['request'],result=ix['record']): raise ValueError('Index/raw mismatch')
        return load_arrays(ix['record'])
