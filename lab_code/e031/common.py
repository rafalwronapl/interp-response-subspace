"""E031 mixed forward/backward budget, guarded resources and durable array cache."""
import os
import time
import uuid
import shutil
import json
import struct
import math
from pathlib import Path
import numpy as np
from research_checks.gate import read
from research_checks.persistence import cached,immutable,request_key
from e012.common import append,rows,save,now,sha
from e024.common import payload,sha_array
from e017.common import memory

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/E031';PREP=ROOT/'analysis/E031_preparation';CONFIG=ROOT/'plans/E031_config.json'


def status(stage,**kw):save(OUT/'STATUS.json',dict(stage=stage,utc=now(),pid=os.getpid(),**kw))


def memory_requirement(root=ROOT):
    path=Path(root)/'models/Qwen3-1.7B-Base/model.safetensors'
    with path.open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size))
    resident=0;peak=0
    for key in sorted(k for k in header if k!='__metadata__'):
        entry=header[key];value=math.prod(entry['shape'])*4;payload_bytes=entry['data_offsets'][1]-entry['data_offsets'][0]
        peak=max(peak,resident+value+payload_bytes);resident+=value
    return dict(FP32_weight_bytes=resident,stream_load_peak_bytes=peak,reserve_bytes=2**29,required_start_bytes=peak+2**29)


def guard(start=False,root=ROOT):
    m=memory();threshold=memory_requirement(root)['required_start_bytes']/2**30 if start else .5
    if min(m['available_gib'],m['commit_headroom_gib'])<threshold:raise RuntimeError('E031 RAM guard before reservation/load')
    if shutil.disk_usage(root).free/2**30<(4 if start else 2):raise RuntimeError('E031 disk guard')
    if (Path(root)/'runs/E031/STOP').exists():raise RuntimeError('STOP requested')
    return m


def donor(d,docs):
    matches=[x for x in docs if (x['group'],x['frame'],x['S'],x['A'])==(d['group'],d['frame'],1-d['S'],d['A'])]
    if len(matches)!=1:raise ValueError('Donor not unique')
    return matches[0]


class Budget:
    def __init__(self,stage,schedule,root=ROOT,resource_guard=None):
        self.stage=stage;self.root=Path(root);self.folder=self.root/'runs/E031';self.folder.mkdir(parents=True,exist_ok=True)
        self.cfg=read(self.root/'plans/E031_config.json')['budget'];self.allowed={o['id']:o['kind'] for o in schedule['operations']}
        self.guard=resource_guard or (lambda:guard(root=self.root))
    def reserve(self,kind,label):
        if self.allowed.get(label)!=kind:raise ValueError('Unscheduled '+kind+': '+label)
        self.guard();rr=list(rows(self.folder/'ledger.jsonl'));res={x['id']:x for x in rr if x['event']=='reserve'};fin={x['id']:x for x in rr if x['event']=='finish'}
        if set(res)!=set(fin):raise RuntimeError('Unfinished reservation requires diagnosis')
        for k in ('forward','backward'):
            if kind==k and (sum(x['kind']==k for x in res.values())>=self.cfg['hard_'+k+'_cap'] or sum(x['kind']==k and x['stage']==self.stage for x in res.values())>=self.cfg['stage_caps'][self.stage][k]):raise RuntimeError(k+' cap')
        elapsed=sum(x['seconds'] for x in fin.values());local=sum(x['seconds'] for k,x in fin.items() if res[k]['stage']==self.stage)
        if elapsed>=self.cfg['max_model_seconds'] or local>=self.cfg['stage_model_seconds_caps'][self.stage]:raise RuntimeError('Persistent model-time cap')
        ticket=uuid.uuid4().hex;append(self.folder/'ledger.jsonl',dict(event='reserve',id=ticket,stage=self.stage,kind=kind,label=label,utc=now()));return ticket
    def finish(self,ticket,ok,seconds):
        rr=list(rows(self.folder/'ledger.jsonl'))
        if sum(x['event']=='reserve' and x['id']==ticket and x['stage']==self.stage for x in rr)!=1 or any(x['event']=='finish' and x['id']==ticket for x in rr):raise ValueError('Invalid finish')
        append(self.folder/'ledger.jsonl',dict(event='finish',id=ticket,ok=bool(ok),seconds=float(seconds),utc=now()))


def ledger_summary(root=ROOT):
    rr=list(rows(Path(root)/'runs/E031/ledger.jsonl'));res={r['id']:r for r in rr if r['event']=='reserve'};fin={r['id']:r for r in rr if r['event']=='finish'}
    if not set(fin)<=set(res):raise ValueError('Orphan finish')
    return dict(attempts=len(res),failed=sum(not x['ok'] for x in fin.values()),unfinished=len(set(res)-set(fin)),model_seconds=sum(x['seconds'] for x in fin.values()),
                kinds={k:sum(x['kind']==k for x in res.values()) for k in ('forward','backward')},
                stages={s:dict(forward=sum(x['stage']==s and x['kind']=='forward' for x in res.values()),backward=sum(x['stage']==s and x['kind']=='backward' for x in res.values()),model_seconds=sum(x['seconds'] for k,x in fin.items() if res[k]['stage']==s)) for s in ('pilot','main')})


class Measurements:
    def __init__(self,subject,stage,root=ROOT):
        self.subject=subject;self.stage=stage;self.root=Path(root);self.out=self.root/'runs/E031'
        self.identity=read(self.root/'analysis/E031_preparation/identity.json')
    def run(self,doc,label,reference=None,**settings):
        reference_id={k:sha_array(np.asarray(v)) for k,v in sorted(reference.items())} if reference is not None else None
        request=dict(model_identity=self.identity['model_identity'],code_identity=self.identity['code_identity'],document=doc,
                     intervention=dict(stage=self.stage,label=label,settings=settings,reference=reference_id),readout=doc.get('readout_ids',[374,525]))
        key=request_key(request);index=self.out/'index'/(label+'.json')
        if index.exists() and payload(index)['request']!=request:raise ValueError('Existing label semantics changed')
        def compute():
            rec,arrays=self.subject.run(doc,label,reference=reference,**settings)
            folder=self.out/'arrays';folder.mkdir(parents=True,exist_ok=True);target=folder/(key+'.npz');temporary=folder/(key+'.tmp.npz')
            np.savez_compressed(temporary,**arrays);os.replace(temporary,target)
            rec['array_file']=target.relative_to(self.root).as_posix();rec['array_sha256']=sha(target)
            return rec
        rec=cached(self.out/'raw',request,compute)
        arrays=self.arrays(rec)
        if not np.isfinite([rec['margin']]+rec['scores']).all():raise ValueError('Nonfinite scalar')
        immutable(index,dict(request_key=key,request=request,result=rec))
        return rec,arrays
    def arrays(self,rec):
        p=self.root/rec['array_file']
        if sha(p)!=rec['array_sha256']:raise ValueError('Array hash changed')
        with np.load(p,allow_pickle=False) as z:
            arrays={k:z[k] for k in z.files}
        if any(v.dtype.kind=='f' and not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite arrays')
        return arrays
    def indexed(self,label):
        ix=payload(self.out/'index'/(label+'.json'));raw=payload(self.out/'raw'/(ix['request_key']+'.json'))
        if raw!=dict(request=ix['request'],result=ix['result']):raise ValueError('Index/raw mismatch')
        return ix['result'],self.arrays(ix['result'])
