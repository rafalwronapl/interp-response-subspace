import os
import time
import uuid
import shutil
from pathlib import Path
import numpy as np
from research_checks.gate import read, sha, digest, create, environment_identity
from research_checks.persistence import immutable, cached, request_key
from e036.common import array_id, save_arrays, load_arrays, append, events, payload
from e017.common import memory

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'analysis/E039_execution'
DATA=ROOT/'analysis/E039_preparation/dataset.json'
OUT=ROOT/'runs/E039'
DOSES=(.5,1.)
METHODS=('C','B0','B1','B2','B3','B4','B5','B6')
CAPS={'pilot':dict(network=736,clt_pair=72,jvp32=60,jvp64=16,local_mlp64=96),
      'main':dict(network=6600,clt_pair=792,jvp32=640,jvp64=0,local_mlp64=0)}
TIMES={'pilot':1800.,'main':10800.}


def guard(start=False):
    m=memory()
    import psutil
    m['process_peak_rss_gib']=getattr(psutil.Process().memory_info(),'peak_wset',int(m['rss_gib']*2**30))/2**30
    if min(m['available_gib'],m['commit_headroom_gib'])<(4 if start else .5):raise RuntimeError('E039 RAM/commit guard')
    if shutil.disk_usage(ROOT).free<(4 if start else 1)*2**30:raise RuntimeError('E039 disk guard')
    if (OUT/'STOP').exists():raise RuntimeError('E039 STOP requested')
    return m


def status(stage,**kw):
    import json
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'STATUS.tmp'
    p.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),**kw)),encoding='utf-8')
    os.replace(p,OUT/'STATUS.json')


def conditions(doc):
    for branch in ('B','C'):
        for i,dose in enumerate(DOSES):
            yield dict(id=doc['id']+'_'+branch+f'_d{i}',branch=branch,dose=dose)


class Budget:
    def __init__(self,stage,folder=None,allowed=None,resource_guard=guard):
        self.stage=stage;self.folder=Path(folder) if folder else OUT/stage
        self.folder.mkdir(parents=True,exist_ok=True);self.log=self.folder/'ledger.jsonl'
        self.allowed=allowed if allowed is not None else read(PREP/(stage+'_operations.json'))['categories']
        self.guard=resource_guard
        ev=events(self.log);r={x['id']:x for x in ev if x['event']=='reserve'};f={x['id']:x for x in ev if x['event']=='finish'}
        self.counts={k:sum(v['category']==k for v in r.values()) for k in CAPS[stage]}
        self.seconds=sum(x['seconds'] for x in f.values());self.failed=sum(not x['ok'] for x in f.values())
        self.unfinished=len(set(r)-set(f))
    def summary(self):return dict(counts=dict(self.counts),seconds=self.seconds,failed=self.failed,unfinished=self.unfinished)
    def call(self,label,category,compute):
        if self.allowed.get(label)!=category:raise ValueError('Unscheduled operation '+label)
        if self.unfinished:raise RuntimeError('Unfinished reservation requires audited recovery')
        if self.counts[category]>=CAPS[self.stage][category] or self.seconds>=TIMES[self.stage]:raise RuntimeError('E039 budget cap')
        resources=self.guard();ticket=uuid.uuid4().hex
        from datetime import datetime,timezone
        append(self.log,dict(event='reserve',id=ticket,label=label,category=category,resources=resources,utc=datetime.now(timezone.utc).isoformat()))
        self.counts[category]+=1;self.unfinished+=1;start=time.monotonic();ok=False
        try:result=compute();ok=True;return result
        finally:
            seconds=time.monotonic()-start
            append(self.log,dict(event='finish',id=ticket,ok=ok,seconds=seconds))
            self.seconds+=seconds;self.failed+=not ok;self.unfinished-=1
            if self.seconds>TIMES[self.stage]:raise RuntimeError('E039 time cap after operation')


class Measurements:
    def __init__(self,stage,budget,identity=None,folder=None):
        self.stage=stage;self.budget=budget;self.folder=Path(folder) if folder else OUT/stage
        self.identity=identity if identity is not None else read(PREP/'identity.json')
        self.forecast_verified=False;self.vector_verified=False
    def run(self,label,category,doc,metadata,compute):
        request=dict(model_identity=self.identity['model_identity'],code_identity=self.identity['code_identity'],document=doc,
            intervention=dict(stage=self.stage,label=label,category=category,metadata=metadata,protocol=self.identity['protocol_identity']),readout=[374,525])
        key=request_key(request);index=self.folder/'index'/(label+'.json')
        if index.exists() and payload(index)['request']!=request:raise ValueError('Changed observation provenance')
        # Check even on resume; an existing target never bypasses forecast integrity.
        if (metadata.get('forecast_readout') or metadata.get('intervention_target')) and not self.vector_verified:
            f=payload(self.folder/'vectors.json');a=load_arrays(f['arrays'])
            n=self.identity.get('row_counts',{}).get(self.stage,48 if self.stage=='pilot' else 576)
            width=self.identity.get('hidden_size',1024)
            if len(f['rows'])!=n or any(a[k].shape!=(n,width) for k in METHODS):
                raise ValueError('Incomplete vector freeze')
            self.vector_verified=True
        if metadata.get('intervention_target') and not self.forecast_verified:
            f=payload(self.folder/'forecasts.json');a=load_arrays(f['arrays'])
            n=self.identity.get('row_counts',{}).get(self.stage,48 if self.stage=='pilot' else 576)
            if len(f['rows'])!=n or set(a)!=set(METHODS) or any(a[k].shape!=(n,) for k in METHODS):
                raise ValueError('Incomplete prospective readout freeze')
            self.forecast_verified=True
        def evaluate():
            arrays=self.budget.call(label,category,compute)
            if any(np.asarray(v).dtype.kind=='f' and not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite measurement')
            if (self.folder/'arrays'/(key+'.npz')).exists():raise ValueError('Orphan array requires audited recovery')
            return save_arrays(self.folder/'arrays',key,arrays)
        record=cached(self.folder/'raw',request,evaluate)
        immutable(index,dict(request=request,record=record,key=key))
        return load_arrays(record)
    def get(self,label):
        ix=payload(self.folder/'index'/(label+'.json'));raw=payload(self.folder/'raw'/(ix['key']+'.json'))
        if raw!=dict(request=ix['request'],result=ix['record']):raise ValueError('Raw/index mismatch')
        return load_arrays(ix['record'])


def freeze(folder,name,arrays,metadata):
    path=folder/(name+'.json')
    if path.exists():
        p=payload(path)
        if {k:v for k,v in p.items() if k!='arrays'}!=metadata or array_id(load_arrays(p['arrays']))!=array_id(arrays):
            raise ValueError('Frozen stage artifact changed')
        return p
    return immutable(path,dict(arrays=save_arrays(folder,name,arrays),**metadata))['payload']
