from pathlib import Path
import json,hashlib,numpy as np
from research_checks.gate import digest
from analysis.A_L21_learned_20261103.common import unwrap,array
B=Path('analysis/A_L21_learned_20261103/stage2');O=B/'run'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
seal=unwrap(O/'completed.json');assert all(sha(O/p)==h for p,h in seal['files'].items());print('sealed',len(seal['files']),flush=True)
s=unwrap(O/'summary.json');f=unwrap(O/'TEST_forecasts.json');fh=sha(O/'TEST_forecasts.json');pred={k:array(v) for k,v in f['arrays'].items()};keyidx={k:i for i,k in enumerate(f['keys'])};U=array(unwrap(B.parent/'stage1_run/MODELS.json')['U']);events=[]
for line in (O/'ledger.jsonl').read_text().splitlines():
 z=json.loads(line);assert digest(z['record'])==z['sha256'];events.append(z['record'])
attempt=[e for e in events if e['event']=='attempt'];ops=[op for e in attempt for op in e['coupled_operations']];assert len(ops)==len(set(ops))==8460;assert set(ops)=={x['id'] for x in json.loads((B/'operations.json').read_text())['operations']}
firsttarget=next(i for i,e in enumerate(attempt) if e['operation'].startswith('LC_') and e['operation'].endswith('_target'));assert sum(e['operation'].endswith('_prefix') for e in attempt[:firsttarget])==1024;assert sum(e['operation'].endswith('_natural') for e in attempt[:firsttarget])==256
cache_count=0
def cached(op):
 global cache_count
 r=unwrap(O/'bundles'/f'{op}.json');c=unwrap(O/'cache'/f"{r['request']}.json");assert digest(c['request'])==r['request'];assert c['request']['intervention']['logical_operations']==r['operations'];cache_count+=1;return c
nat={};rows=[];reconstructed=0
for path in sorted((O/'rows').glob('*.json')):
 r=unwrap(path);id=r['id'];docid=id.rsplit('_',2)[0]
 if docid not in nat:nat[docid]=cached(docid+'_natural')['result']['state']
 n=nat[docid];tc=cached(id+'_target');assert tc['request']['intervention']['forecasts_sha256']==fh;t=tc['result']['state'];D=array(t['y'])-array(n['y']);truth=float(array(t['logits'])[1])-float(array(t['logits'])[0]);natural=float(array(n['logits'])[1])-float(array(n['logits'])[0]);assert r['truth']==truth and r['natural_margin']==natural and r['margins']['zero_change']==natural
 v={k:a[keyidx[id]] for k,a in pred.items()};v['oracle_inside']=v['P']+U@(U.T@(D.astype(float)-v['P']));v['zero_change']=np.zeros_like(D,dtype=float)
 tail=cached(id+'_P');assert tail['request']['intervention']['forecasts_sha256']==fh;names=[op[len(id)+1:] for op in tail['request']['intervention']['logical_operations']];assert names==['P','C','same_U','Direct384','mean8','oracle_inside']
 for a,margin in zip(names,tail['result']):
  assert margin==r['margins'][a];np.testing.assert_array_equal(np.asarray(array(n['y']).astype(float)+v[a],np.float32),np.asarray(tail['request']['intervention']['targets'][a],np.float32))
 for a,x in v.items():
  assert abs(float(np.sum((x-D)**2))-r['vector_L2'][a])<1e-10;assert abs(float(np.sum(((x-D)@U)**2))-r['coefficient_L2'][a])<1e-10;reconstructed+=2
 rows.append(r)
print('rawrows',len(rows),flush=True)
blocks=sorted({r['block'] for r in rows});assert len(rows)==1024 and len(blocks)==64;metricchecks=0
for arm,m in s['metrics'].items():
 loss=np.array([np.mean([(r['margins'][arm]-r['truth'])**2 for r in rows if r['block']==bb]) for bb in blocks]);np.testing.assert_allclose(loss,m['block_MSE'],atol=1e-14,rtol=0);assert abs(loss.mean()-m['MSE'])<1e-14
 for k in ['vector_L2','coefficient_L2']:assert abs(np.mean([r[k][arm] for r in rows])-m[k])<1e-11
 metricchecks+=67
for name,c in s['contrasts'].items():
 a,ref=name.split('-');diff=np.array(s['metrics'][a]['block_MSE'])-np.array(s['metrics'][ref]['block_MSE']);assert abs(diff.mean()-c['delta_MSE'])<1e-14
for key,g in s['subgroups'].items():
 field,val=key.split('=');rr=[r for r in rows if str(r[field])==val];assert len(rr)==g['conditions']
 for a,v in g['MSE'].items():assert abs(np.mean([(r['margins'][a]-r['truth'])**2 for r in rr])-v)<1e-14
assert seal['attempts']==8460 and seal['aliases']==0 and seal['backwards']==0 and seal['model_load_attempts']==1
out=dict(reviewer='/root/astra_learned_plan_review',sealed_files=len(seal['files']),ledger_attempt_bundles=len(attempt),unique_logical_forwards=len(ops),all_prefixes_and_naturals_before_TEST_targets=True,raw_rows_reconstructed=len(rows),native_targets_and_margins_reconstructed=True,all_frozen_patch_targets_verified=True,vector_coefficient_checks=reconstructed,aggregate_block_checks=metricchecks,subgroups_and_paired_deltas_checked=True,blocks=len(blocks),qualification=[unwrap(p) for p in sorted((O/'qualification').glob('*.json'))],metrics={a:m['MSE'] for a,m in s['metrics'].items()},C_reduction_percent=100*(1-s['metrics']['C']['MSE']/s['metrics']['P']['MSE']),C_over_Direct_percent=100*(s['metrics']['C']['MSE']/s['metrics']['Direct384']['MSE']-1),native_calls=0,refits=0,bootstrap_reruns=0)
(B/'review/independent_completion_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
