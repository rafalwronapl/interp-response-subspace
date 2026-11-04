"""Independent prefix-once durable head audit; never re-evaluates a head."""
import json,hashlib,numpy as np
from pathlib import Path
from research_checks.gate import digest
from research_checks.persistence import request_key
from analysis.C2_fresh_fit_holdout_20261102.worker import ROOT,read,sha
BASE=ROOT/'analysis/A_gemma270_extension_20261102';stage=BASE/'evaluation_v1';cfg=read(stage/'config.json');out=ROOT/cfg['output'];docs=read(ROOT/cfg['corpus'])['documents'];expected={d['id']+'__'+arm:(d,arm) for d in docs for arm in cfg['arms']};progress=BASE/'head_peer_checks_qwen_progress_v1.json'
state=read(progress) if progress.exists() else dict(checked=[],files={},contexts={},scope='Immutableprefix audit only, no head/model replay')
def load(p):
 blob=p.read_bytes();v=json.loads(blob);assert digest(v['payload'])==v['payload_sha256'];state['files'][p.relative_to(ROOT).as_posix()]=hashlib.sha256(blob).hexdigest();return v['payload']
identity=load(out/'identity.json');contract=read(stage/'contract.json');actual=dict(model=sha(ROOT/read(ROOT/cfg['capture_config'])['model_identity']),code=digest(dict(config=sha(stage/'config.json'),inputs=sha(stage/'input_manifest.json'),sources={n:sha(ROOT/n) for n in contract['tracked_files'] if n.endswith('.py')})));assert identity==actual
seen=set(state['checked'])
for p in (out/'completed').glob('*.json'):
 op=p.stem
 if op in seen:continue
 assert op in expected;d,arm=expected[op];doc=d['id'];r=load(p)
 if doc not in state['contexts']:
  c=load(out/'contexts'/(doc+'.json'));assert c['document']==doc and c['fixed_pairs']==cfg['fixed_pairs'];state['contexts'][doc]=c['context_sha256']
 request=dict(model_identity=identity['model'],code_identity=identity['code'],document=dict(document=d,context_sha256=state['contexts'][doc]),intervention=dict(arm=arm),readout=dict(answer_ids=[563,659],orientation='is-minus-are'));key=request_key(request);assert r==dict(operation=op,request=key);cached=load(out/'cache'/(key+'.json'));assert cached['request']==request and np.ndim(cached['result'])==0 and np.isfinite(cached['result']);seen.add(op)
state['checked']=sorted(seen);state['checked_count']=len(seen);state['identity']=identity;state['model_calls']=0
progress.write_text(json.dumps(state,indent=2)+'\n');print('Incremental independentGemmahead checks',len(seen),'/20480;0headreplays')
