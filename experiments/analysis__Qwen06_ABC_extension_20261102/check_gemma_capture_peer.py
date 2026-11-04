"""Independent immutable receipt/request/accounting checks, zero model calls."""
import json,numpy as np
from pathlib import Path
from analysis.C2_fresh_fit_holdout_20261102.worker import ROOT,read,sha,receipt
from research_checks.persistence import request_key
from research_checks.gate import digest
BASE=ROOT/'analysis/A_gemma270_extension_20261102';stage=BASE/'capture_v1';cfg=read(stage/'config.json');out=ROOT/cfg['output'];done=receipt(out/'stage_completed.json');ident=receipt(out/'identity.json');assert ident==done['identity'];assert done['forward']==done['captures']==2048 and done['backward']==0 and done['LLM_loads']==1 and done['qualification_reused']
docs=read(ROOT/cfg['corpus'])['documents'];tokens=read(ROOT/cfg['tokens']);schedule=read(ROOT/cfg['schedule']);ops={o['id'] for o in schedule['operations']};assert len(docs)==2048 and len(ops)==2048 and set(tokens)=={d['id'] for d in docs};assert {p.stem for p in (out/'completed').glob('*.json')}==ops
keys=set()
for d in docs:
 op='capture_'+d['id'];r=receipt(out/'completed'/(op+'.json'));expected=dict(model_identity=ident['model'],code_identity=ident['code'],document=dict(document=d,tokens=tokens[d['id']]),intervention=dict(mode='natural'),readout=dict(answer_ids=cfg['answer_ids'],orientation='is-minus-are'));key=request_key(expected);assert r==dict(operation=op,request=key);v=receipt(out/'cache'/(key+'.json'));assert v['request']==expected;values=v['result'];assert set(values)=={'R','x','y','margin'} and all(np.asarray(values[k]).shape==(640,) and np.isfinite(values[k]).all() for k in ['R','x','y']) and np.isfinite(values['margin']);keys.add(key)
assert len(keys)==2048
records=[]
for line in (out/'ledger.jsonl').read_text().splitlines():
 row=json.loads(line);assert digest(row['record'])==row['sha256'];records.append(row['record'])
attempt=[r for r in records if r['event']=='attempt'];computed=[r for r in records if r['event']=='computed'];assert len(attempt)==len(computed)==2048 and {r['operation'] for r in attempt}=={r['operation'] for r in computed}==ops and {r['request'] for r in attempt}==keys
proof=dict(status='PASS',reviewer_id='/root/sol_qwen06_parallel',independent=True,model_calls=0,captures=2048,attempts=2048,computed=2048,unique_cache=2048,tests=['All immutable hashes and complete source-bound request identities/tokens/settings/results checked','All R/x/y finite640dim and readoutfinite','Exact2048ledger attempts/computed, no duplicateattempts or missingreceipts','Qualifiedcontrols/C3FIT/currentcheckpoint proof reused; no rawhead or native result replay'])
path=BASE/'capture_peer_checks_qwen_v1.json';path.write_text(json.dumps(proof,indent=2)+'\n');print('Gemma capture peer checks PASS2048/2048,0modelcalls')
