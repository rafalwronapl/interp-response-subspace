"""Offline completion acceptance packet; no fitting, model calls or resampling."""
from pathlib import Path
import json,numpy as np
from research_checks.gate import sha,digest,read,create,snapshot
BASE=Path(__file__).parent;ROOT=BASE.parents[2];REL=BASE.relative_to(ROOT).as_posix()
def unwrap(p):
 v=read(p);assert digest(v['payload'])==v['payload_sha256'];return v['payload']
def main():
 out=BASE/'run';done=unwrap(out/'completed.json');assert done['complete'] and read(BASE/'launch_exit.json')['returncode']==0
 files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['completed.json','STATUS.json','STATUS.tmp']};assert files==done['files']
 rows=[unwrap(p) for p in sorted((out/'rows').glob('*.json'))];assert len(rows)==1024 and len({r['block'] for r in rows})==64
 for b in {r['block'] for r in rows}:assert len([r for r in rows if r['block']==b])==16
 s=unwrap(out/'summary.json')
 for arm,m in s['metrics'].items():
  block=np.array([np.mean([(r['margins'][arm]-r['truth'])**2 for r in rows if r['block']==b]) for b in sorted({r['block'] for r in rows})]);np.testing.assert_allclose(block,m['block_MSE'],atol=1e-12,rtol=1e-12);assert abs(block.mean()-m['MSE'])<1e-12
  for metric in ['vector_L2','coefficient_L2']:assert abs(np.mean([r[metric][arm] for r in rows])-m[metric])<1e-10
 records=[json.loads(x) for x in (out/'ledger.jsonl').read_text().splitlines()]
 for x in records:assert digest(x['record'])==x['sha256']
 attempts=[x['record'] for x in records if x['record']['event']=='attempt'];ids=[i for x in attempts for i in x['coupled_operations']];aliases={p.stem for p in (out/'aliases').glob('*.json')};planned={x['id'] for x in read(BASE/'operations.json')['operations']}
 assert len(ids)==len(set(ids))==done['attempts'] and not set(ids)&aliases and set(ids)|aliases==planned and len(planned)==8460
 controls=[unwrap(p) for p in (out/'qualification').glob('*.json')];assert len(controls)==2 and all(x['passed'] and x['maxabs']<=1e-4 for x in controls)
 expected=sha(out/'TEST_forecasts.json');targets=0
 for x in attempts:
  if not x['operation'].endswith('_target'):continue
  request=unwrap(out/'cache'/(x['request']+'.json'))['request'];assert request['intervention']['forecasts_sha256']==expected;targets+=1
 assert targets==1024 and done['backwards']==0
 checks=dict(passed=True,files=len(files),rows=1024,blocks=64,arms=len(s['metrics']),logical_forward=done['attempts'],aliases=len(aliases),physical_forward=done['physical_forward_calls'],backward=0,model_load_attempts=done['model_load_attempts'],qualification_max=max(x['maxabs'] for x in controls),forecast_binding_all_targets=True,no_model_fit_bootstrap=True)
 create(BASE/'completion_checks.json',checks)
 inputs=[out/p for p in files]+[out/'completed.json',BASE/'REPORT.txt',BASE/'launch_exit.json',BASE/'pre_run_snapshot.json',BASE/'pre_run_review.json',BASE/'completion_checks.json']
 create(BASE/'completion_manifest.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in inputs}))
 create(BASE/'completion_contract.json',dict(schema_version=1,id='A_L21_LEARNED_FRESH64_COMPLETION',kind='maintenance',primary_agent='/root/a_l21_five_campaign',purpose='Actual prospective learned correction/readout completion; preserve all baselines and outcomes',tracked_files=[REL+'/complete.py'],tracked_trees=[],input_manifests=[REL+'/completion_manifest.json'],gates=[dict(id=k,category=k,evidence=REL+'/completion_checks.json',pointer='/passed',op='eq',expected=True,reason='Immutableoutputs/rowmeans/ledger/qualification andforecast provenance verified offline') for k in ['hooks','data','pilot','accounting','restart']],circuits=[],structural_scope='No new measurements; exact nativeL21 intervention/readout stage complete',entrypoint=None,schedule=None,budget=None))
 create(BASE/'completion_snapshot.json',snapshot(ROOT,REL+'/completion_contract.json','completion'));print(checks)
if __name__=='__main__':main()
