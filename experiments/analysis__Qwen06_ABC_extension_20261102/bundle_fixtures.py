"""Crash/accounting/resume/budget fixtures for new stage-local BundleRuntime."""
import tempfile,json
from pathlib import Path
from analysis.Qwen06_ABC_extension_20261102.bundle_runtime import BundleRuntime as Runtime
from analysis.C2_fresh_fit_holdout_20261102.worker import sha,receipt
BASE=Path(__file__).parent
def main():
 ops=['o'+str(i) for i in range(50)];schedule=dict(operations=[dict(id=o,kind='forward') for o in ops]);identity=dict(model='synthetic',code='bundle-fixture');kwargs=dict(document=dict(id='d'),intervention=dict(arms=ops),readout=dict(answer_ids=[374,525]))
 with tempfile.TemporaryDirectory() as tmp:
  path=Path(tmp);calls=[]
  def callback():calls.append(1);return {op:float(i) for i,op in enumerate(ops)}
  rt=Runtime(path,identity,schedule);first=rt.compute(ops[0],**kwargs,function=callback,extra_operations=ops[1:]);assert len(rt.attempts)==50 and len(rt.records)==2;assert rt.records[0]['event']=='attempt' and rt.records[0]['logical_forward']==rt.records[1]['logical_forward']==50
  assert len(list((path/'cache').glob('*.json')))==len(list((path/'bundles').glob('*.json')))==1
  bundle=path/'bundles/o0.json';payload=receipt(bundle);assert payload['operations']==ops and payload['logical_forward']==50;h=sha(bundle);ledger=sha(path/'ledger.jsonl');rt=Runtime(path,identity,schedule)
  assert rt.compute(ops[0],**kwargs,function=callback,extra_operations=ops[1:])==first and len(calls)==1 and sha(bundle)==h and sha(path/'ledger.jsonl')==ledger
  bundle.unlink();rt=Runtime(path,identity,schedule);assert rt.compute(ops[0],**kwargs,function=callback,extra_operations=ops[1:])==first and len(calls)==1 and receipt(bundle)==payload and sha(path/'ledger.jsonl')==ledger
  for extra in [[ops[0]],['UNKNOWN']]:
   try:rt.compute(ops[0],**kwargs,function=callback,extra_operations=extra)
   except ValueError:pass
   else:raise AssertionError('Invalid batch accepted')
  try:rt.compute(ops[0],document=dict(id='changed'),intervention=kwargs['intervention'],readout=kwargs['readout'],function=callback,extra_operations=ops[1:])
  except ValueError:pass
  else:raise AssertionError('Consumed overlap accepted')
  assert len(calls)==1 and sha(path/'ledger.jsonl')==ledger
 with tempfile.TemporaryDirectory() as tmp:
  path=Path(tmp);calls=[];rt=Runtime(path,identity,schedule)
  def crash():calls.append(1);raise RuntimeError('Synthetic interruption before durable cache')
  try:rt.compute(ops[0],**kwargs,function=crash,extra_operations=ops[1:])
  except RuntimeError:pass
  else:raise AssertionError('No interruption')
  assert len(rt.attempts)==50 and len(rt.records)==1;rt=Runtime(path,identity,schedule)
  try:rt.compute(ops[0],**kwargs,function=crash,extra_operations=ops[1:])
  except ValueError:pass
  else:raise AssertionError('Hidden retry')
  assert len(calls)==1 and len(rt.records)==1
 proof=dict(status='PASS',pretrained_calls=0,tests=['50unique scheduledIDs durablyreservedBEFOREcallback andfullycharged','Onecache+bundlereceipt with50logicalops','Cachedresume zero callback/ledger changes; byteidentitypreserved','Missingreceipt repairedfromvalidcache zero calls','Duplicate/unknown/overlap rejected','Partialfailure consumesall50, reviewedretryrequired'],source_hashes={n:sha(BASE/n) for n in ['bundle_runtime.py','bundle_fixtures.py']})
 (BASE/'bundle_fixture_v1.json').write_text(json.dumps(proof,indent=2)+'\n');print('BundleRuntime6focusedfixturesPASS;0modelcalls')
if __name__=='__main__':main()
