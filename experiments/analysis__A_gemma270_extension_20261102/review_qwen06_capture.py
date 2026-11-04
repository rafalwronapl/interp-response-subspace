"""Independent incremental Q06 capture audit; saved bytes only, no model imports."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'analysis/Qwen06_ABC_extension_20261102';ST=B/'capture_base_v2';DEST=B/'review_sol_gemma_peer'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def payload(p):
 data=p.read_bytes();v=json.loads(data);assert digest(v['payload'])==v['payload_sha256'],str(p);return v['payload'],hashlib.sha256(data).hexdigest()
def run():
 DEST.mkdir(exist_ok=True);cfg=read(ST/'config.json');out=ROOT/cfg['output'];identity,_=payload(out/'identity.json');pre=read(ST/'pre_run_snapshot_v3.json');assert pre['snapshot_sha256']=='048e4458d6b98a5644aa5a30b66c82a88d40df86afc8906c56990b57bff4b1da';assert digest(pre['payload'])==pre['snapshot_sha256'];tokens=read(ROOT/cfg['tokens']);docs={d['id']:d for d in read(ROOT/cfg['corpus'])['documents']};ops={x['id'] for x in read(ROOT/cfg['schedule'])['operations']};index=DEST/'capture_index_v1.json';proof=read(index) if index.exists() else dict(identity=identity,records={});assert proof['identity']==identity
 for cp in (out/'completed').glob('*.json'):
  op=cp.stem
  if op in proof['records']:continue
  assert op in ops;c,ch=payload(cp);assert c['operation']==op;cache=out/'cache'/(c['request']+'.json');v,vh=payload(cache);request=v['request'];assert digest(request)==c['request'];assert request['model_identity']==identity['model'] and request['code_identity']==identity['code'];d=request['document']['document'];assert d==docs[d['id']] and request['document']['tokens']==tokens[d['id']];assert request['readout']==dict(answer_ids=cfg['assets']['answer_ids'],orientation='is-minus-are')
  if op.startswith('capture_'):
   assert op=='capture_'+d['id'] and request['intervention']==dict(mode='natural');result=v['result'];assert set(result)=={'R','x','y','margin'}
   for k in ['R','x','y']:a=np.asarray(result[k],float);assert a.shape==(1024,) and np.isfinite(a).all()
   assert np.isfinite(result['margin'])
  proof['records'][op]=dict(request=c['request'],receipt=cp.relative_to(ROOT).as_posix(),receipt_sha256=ch,cache=cache.relative_to(ROOT).as_posix(),cache_sha256=vh)
 with index.open('w',encoding='utf-8') as f:json.dump(proof,f,allow_nan=False)
 print('Q06 independently cached immutable proofs',len(proof['records']),'/',len(ops),flush=True)
if __name__=='__main__':run()
