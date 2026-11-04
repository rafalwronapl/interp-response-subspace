"""Independent actual native-readout audit; no model loading or new estimands."""
import base64,hashlib,json
from pathlib import Path
from datetime import datetime
import numpy as np
from threadpoolctl import threadpool_limits
from research_checks.gate import digest

ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'analysis/AB_budget_readout_20261103';OUT=BASE/'run'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def unwrap(p):
    v=read(p);assert digest(v['payload'])==v['payload_sha256'],str(p);return v['payload']
def array(v):return np.frombuffer(base64.b64decode(v['data']),dtype=v['dtype']).reshape(v['shape']).copy()
snapshot=read(BASE/'completion_snapshot.json');assert digest(snapshot['payload'])==snapshot['snapshot_sha256']
for p,h in snapshot['payload']['files'].items():assert sha(ROOT/p)==h,p
done=unwrap(OUT/'completed.json');summary=unwrap(OUT/'summary.json')
rows={p.stem:unwrap(p) for p in (OUT/'rows').glob('*.json')};assert len(rows)==1024
blocks=sorted({r['block'] for r in rows.values()});assert len(blocks)==64
assert all(sum(r['block']==b for r in rows.values())==16 for b in blocks)
arms=list(summary['metrics']);assert len(arms)==17
loss={}
for a in arms:
    values=np.array([np.mean([(r['margins'][a]-r['truth'])**2 for r in rows.values() if r['block']==b]) for b in blocks]);loss[a]=values
    np.testing.assert_allclose(values,summary['metrics'][a]['block_MSE'],atol=1e-15)
    assert np.isclose(values.mean(),summary['metrics'][a]['MSE'],rtol=1e-13)
    for metric in ['vector_L2','coefficient_L2']:
        assert np.isclose(np.mean([r[metric][a] for r in rows.values()]),summary['metrics'][a][metric],rtol=1e-13)
    assert np.isclose(np.mean([r['margins'][a]-r['truth'] for r in rows.values()]),summary['metrics'][a]['signed_bias'])
diff=loss['C_32']-loss['Direct_32']
# Reproduce exactly the prespecified saved bootstrap stream, no new inference.
draw=np.random.default_rng(3111033).integers(0,64,(1999,64));ci=np.quantile(diff[draw].mean(1),[.025,.975])
assert np.isclose(diff.mean(),summary['primary']['delta_MSE'],rtol=1e-13)
np.testing.assert_array_equal(ci,summary['primary']['nominal95'])
forecast_envelope=read(OUT/'TEST_forecasts.json');forecast_time=datetime.fromisoformat(forecast_envelope['created_utc'])
forecasts={a:array(v) for a,v in forecast_envelope['payload']['arrays'].items()}
keyindex={key:i for i,key in enumerate(forecast_envelope['payload']['keys'])};del forecast_envelope
assert len(forecasts)==16 and len(keyindex)==1024
freeze_sha=sha(OUT/'TEST_forecasts.json');U=array(unwrap(BASE/'MODELS.json')['U'])
ledger=[read_line for read_line in (json.loads(line) for line in (OUT/'ledger.jsonl').read_text().splitlines())]
for rec in ledger:assert digest(rec['record'])==rec['sha256']
attempts=[v['record'] for v in ledger if v['record']['event']=='attempt'];logical=[op for v in attempts for op in v['coupled_operations']]
assert len(attempts)==3336 and len(logical)==len(set(logical))==18726
planned={v['id'] for v in read(BASE/'operations.json')['operations']};assert set(logical)==planned
assert done['attempts']==18726 and done['physical_forward_calls']==3336 and done['backwards']==0
assert len(list((OUT/'loads').glob('*.json')))==1 and not list((OUT/'aliases').glob('*.json'))
natural={};target_count=0;tail_count=0;prefix_count=0
with threadpool_limits(limits=2):
    for attempt in attempts:
        cache=unwrap(OUT/'cache'/(attempt['request']+'.json'));q=cache['request'];res=cache['result'];setting=q['intervention'];role=setting['role']
        assert digest(q)==attempt['request'] and setting['logical_operations']==attempt['coupled_operations']
        doc=q['document'];stamp=datetime.fromisoformat(attempt['utc'])
        if role=='natural':
            y=array(res['state']['y']);lg=array(res['state']['logits']);natural[doc['id']]=(y,float(lg[1])-float(lg[0]))
            assert stamp<forecast_time
        elif role=='target_free_prefix':
            assert not {'y','Y','logits'}&set(res);assert stamp<forecast_time;prefix_count+=1
        elif role=='TEST_native_target':
            assert stamp>forecast_time and setting['forecasts_sha256']==freeze_sha
            key=attempt['operation'][:-len('_target')];r=rows[key];lg=array(res['state']['logits']);y=array(res['state']['y'])
            assert r['truth']==float(lg[1])-float(lg[0]) and r['natural_margin']==natural[doc['id']][1]
            D=y-natural[doc['id']][0];ix=keyindex[key]
            for arm in arms:
                v=np.zeros_like(D,dtype=float) if arm=='zero_change' else forecasts[arm][ix]
                assert np.isclose(np.sum((v-D)**2),r['vector_L2'][arm],rtol=1e-12,atol=1e-12)
                assert np.isclose(np.sum(((v-D)@U)**2),r['coefficient_L2'][arm],rtol=1e-12,atol=1e-12)
            target_count+=1
        elif role=='batched_suffix':
            assert setting['forecasts_sha256']==freeze_sha
            key=doc['id']+'_'+setting['branch']+'_'+str(setting['dose']);r=rows[key];ix=keyindex[key]
            assert len(attempt['coupled_operations'])==len(res)==16
            for op,value in zip(attempt['coupled_operations'],res):
                arm=op[len(key)+1:];assert r['margins'][arm]==value
                expected=(natural[doc['id']][0].astype(float)+forecasts[arm][ix]).astype(np.float32)
                np.testing.assert_array_equal(expected,np.asarray(setting['targets'][arm],np.float32))
            tail_count+=1
assert target_count==prefix_count==tail_count==1024 and len(natural)==256
quals=[unwrap(p) for p in (OUT/'qualification').glob('*.json')];assert len(quals)==2
assert all(v['batch_size']==16 and v['maxabs']<=1e-4 and v['passed'] for v in quals)
assert all(summary['metrics'][f'Direct_{n}']['MSE']<summary['metrics'][f'C_{n}']['MSE'] for n in [64,128,256,384])
assert summary['metrics']['C_64']['MSE']>summary['metrics']['C_32']['MSE']
result=dict(passed=True,snapshot_sha256=snapshot['snapshot_sha256'],snapshot_files=len(snapshot['payload']['files']),
    raw_native_targets_checked=target_count,raw_suffix_batches_checked=tail_count,prefixes_checked=prefix_count,
    native_rows1024_blocks64=True,all17metrics_recomputed=True,all_vector_and_coefficient_errors_recomputed=True,
    all16384suffix_targets_and_margins_checked=True,global_forecast_chronology=True,
    logical_forward=18726,physical_forward=3336,model_loads=1,backwards=0,
    primary_delta_MSE=float(diff.mean()),exact_prespecified_bootstrap95=ci.tolist(),
    retained_other_budget_losses_and_nonmonotonicity=True,native_calls_by_reviewer=0)
Path(__file__).with_name('independent_completion_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
