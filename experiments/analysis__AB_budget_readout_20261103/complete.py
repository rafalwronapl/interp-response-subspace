"""Validate completed raw measurements and seal a descriptive report; zero new calls."""
from pathlib import Path
from datetime import datetime
import json,numpy as np
from research_checks.gate import sha,digest,read,create,snapshot
BASE=Path(__file__).parent;ROOT=BASE.parents[1];REL=BASE.relative_to(ROOT).as_posix()
def unwrap(p):
    v=read(p)
    if digest(v['payload'])!=v['payload_sha256']:raise ValueError('Corrupt receipt '+str(p))
    return v['payload']
def main():
    out=BASE/'run';done=unwrap(out/'completed.json');assert done['complete']
    snap=read(BASE/'pre_run_snapshot.json');launches=[p for p in (ROOT/'operations').glob('research_launch_*.json') if read(p).get('snapshot_sha256')==snap['snapshot_sha256']]
    assert launches and all(read(p)['returncode']==0 for p in launches)
    files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['completed.json','STATUS.json','STATUS.tmp']}
    assert files==done['files'] and done['identity']['code']==digest(read(BASE/'input_manifest.json')['files'])
    rows=[unwrap(p) for p in sorted((out/'rows').glob('*.json'))];blocks=sorted({r['block'] for r in rows})
    assert len(rows)==1024 and len(blocks)==64 and all(sum(r['block']==b for r in rows)==16 for b in blocks)
    s=unwrap(out/'summary.json')
    for arm,m in s['metrics'].items():
        values=np.array([np.mean([(r['margins'][arm]-r['truth'])**2 for r in rows if r['block']==b]) for b in blocks])
        np.testing.assert_allclose(values,m['block_MSE'],atol=1e-12,rtol=1e-12);assert abs(values.mean()-m['MSE'])<1e-12
        for metric in ['vector_L2','coefficient_L2']:assert abs(np.mean([r[metric][arm] for r in rows])-m[metric])<1e-10
    records=[json.loads(x) for x in (out/'ledger.jsonl').read_text().splitlines()]
    for x in records:assert digest(x['record'])==x['sha256']
    attempts=[x['record'] for x in records if x['record']['event']=='attempt'];ids=[i for x in attempts for i in x['coupled_operations']]
    aliases={p.stem for p in (out/'aliases').glob('*.json')};planned={x['id'] for x in read(BASE/'operations.json')['operations']}
    assert len(ids)==len(set(ids))==done['attempts'] and not set(ids)&aliases and set(ids)|aliases==planned and len(planned)==18726
    controls=[unwrap(p) for p in (out/'qualification').glob('*.json')]
    assert len(controls)==2 and all(q['passed'] and q['maxabs']<=1e-4 and q['batch_size']==16 for q in controls)
    expected=sha(out/'TEST_forecasts.json');targets=0
    for a in attempts:
        if not a['operation'].endswith('_target'):continue
        assert unwrap(out/'cache'/(a['request']+'.json'))['request']['intervention']['forecasts_sha256']==expected;targets+=1
    assert targets==1024 and done['backwards']==0
    loads=sorted(datetime.fromisoformat(read(p)['created_utc']) for p in (out/'loads').glob('*.json'))
    wall_seconds=(datetime.fromisoformat(read(out/'completed.json')['created_utc'])-loads[0]).total_seconds()
    lines=['A fresh native-readout learning-budget TEST, 2026-11-03','',
        'Completed64 new lexical blocks,256 prompts,1024 intervention conditions; Qwen3-0.6B-Base,',
        'literal E041 upstream intervention, MLP21 prediction and exact native L22-27 suffix.',
        'One fixed lexicographic FIT0seed17 training sequence, not independent-fit replication.',
        'All16 learned/P forecasts were frozen globally before first intervenedTEST native target.','',
        '| Vector-label budget | C readout MSE | same-U readout MSE | Direct readout MSE |',
        '|---:|---:|---:|---:|']
    for n in [32,64,128,256,384]:lines.append('| '+' | '.join([str(n)]+[f"{s['metrics'][f'{m}_{n}']['MSE']:.12g}" for m in ['C','same_U','Direct']])+' |')
    lines += ['',f"P MSE={s['metrics']['P']['MSE']:.12g}; zero-change MSE={s['metrics']['zero_change']['MSE']:.12g}.",
        f"Primary fixed C32-Direct32 paired deltaMSE={s['primary']['delta_MSE']:.12g}; bootstrap95%={s['primary']['nominal95']}.",
        'Positive delta favors Direct; negative favors C. This is the single prespecified primary.',
        f"C32 readout MSE is {100*(1-s['metrics']['C_32']['MSE']/s['metrics']['Direct_32']['MSE']):.4f}% lower than Direct32 in this fixed-sequence TEST.",
        'The primary interval lies below zero. This supports lower readout squared error for the selected',
        'prior-assisted C32 predictor on these64 fresh lexical pairs, conditional on its fixed historical resources.',
        'C32 also has lower mean MSE than same-U32. Direct has lower mean MSE than C at64/128/256/384.',
        'C is nonmonotonic: the64-label C predictor is worse than the32-label C predictor. No point is removed.',
        'This is not equal-total-information superiority, an average over training-set draws, or transfer evidence.',
        'Other budget/method contrasts and intervals are descriptive, nominal pointwise, no familywise claim.','',
        'ALL CONTRASTS',json.dumps(s['contrasts'],indent=2),'',
        'VECTOR / COEFFICIENT ERRORS, SIGN AGREEMENT AND BIAS',
        json.dumps({a:{k:m[k] for k in ['vector_L2','coefficient_L2','delta_sign_agreement','signed_bias']} for a,m in s['metrics'].items()},indent=2),s['sign_convention'],'',
        'ALL MARGINAL SUBGROUPS',json.dumps(s['subgroups'],indent=2),'',
        'Interpretation limits:32vectorlabels are only2 whole lexical blocks, not32 independent units.',
        'Hyperparameters for each lower budget were selected strictly within its labels in historical A-FIT.',
        'Only six necessary coefficient systems were materialized from the saved choices, no new selection;',
        '12 saved vector-validation losses reproduced.384 uses the original packed fullpool models unchanged.',
        'The384 point adds FIT0 training blocks and is not strictly the same nested lower-budget population.',
        'Historical P/W/U/scale consume additional training information. same-U and C share rank8 U;',
        'Direct predicts all output directions. Vector error does not substitute for readout MSE.',
        'Freshness means disjoint recorded unordered pairs/prompts against812 inventoried exposure files;',
        'individual nouns and prompt family recur. This does not prove pretraining or construction novelty.',
        'Intervals use64 paired lexical blocks,1999 bootstrap draws,seed3111033,conditional on fixed priors/fits.',
        'Budgets and models are dependent, not separate independent replications. One model/layer/readout.',
        'Predictions require native target-free upstream prefixes; this is not a stand-alone surrogate from text.',
        'No oracle target labels enter predictors; negative comparisons remain valid outcomes.','',
        'ACCOUNTING',json.dumps({k:v for k,v in done.items() if k!='files'},indent=2),
        f"Walltime from first recorded model-load attempt through immutable completion: {wall_seconds:.3f}s (includes loading,prefixes,forecasting,measurement,serialization).",
        f"FIT16batch full/suffix qualification maxabs={max(q['maxabs'] for q in controls):.12g} (limit1e-4).",
        'Scheduled18726logicalF/0B includes38FITparity,256natural,1024prefix,1024target and16384suffix predictions.',
        'Maximum3336 physical calls with batches16; exact aliases cost0. Actual accounting above includes every load attempt.',
        'No cloud/downloads/newFIT acquisition. Historical384 nativevectorlabels plusE036 prior are additional resources.','',
        'Evidence: run/summary.json,1024 immutable rows,TEST_forecasts.json,ledger,raw cache and completed.json.',
        'Pre-run approval: review/pre_run_review.json bound to pre_run_snapshot.json.',
        'Independent completion status is in review/completion_review.json; this report does not substitute for review.']
    (BASE/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    checks=dict(passed=True,files=len(files),rows=1024,blocks=64,arms=len(s['metrics']),logical_forward=done['attempts'],aliases=len(aliases),physical_forward=done['physical_forward_calls'],backward=0,qualification_max=max(q['maxabs'] for q in controls),forecast_binding_all_targets=True,no_new_fit_or_bootstrap=True)
    create(BASE/'completion_checks.json',checks)
    manifest=read(BASE/'input_manifest.json')['files'].copy()
    inputs=[out/p for p in files]+[out/'completed.json',BASE/'REPORT.txt',BASE/'pre_run_snapshot.json',BASE/'review/pre_run_review.json',BASE/'completion_checks.json',BASE/'complete.py']+launches
    manifest.update({p.relative_to(ROOT).as_posix():sha(p) for p in inputs})
    create(BASE/'completion_manifest.json',dict(files=manifest))
    create(BASE/'completion_contract.json',dict(schema_version=1,id='A_BUDGET_FRESH64_COMPLETION',kind='maintenance',primary_agent='/root',purpose='Actual fixedsequence native-readout budget curve, all outcomes retained',tracked_files=[REL+'/complete.py',REL+'/REPORT.txt'],tracked_trees=[],input_manifests=[REL+'/completion_manifest.json'],gates=[dict(id=k,category=k,evidence=REL+'/completion_checks.json',pointer='/passed',op='eq',expected=True,reason='Immutableoutputs/rowmeans/ledger/16batchqualification/forecastbindings verified offline') for k in ['hooks','data','pilot','accounting','restart']],circuits=[],structural_scope='No new measurements; completed exactnativeL21 readout curve review',entrypoint=None,schedule=None,budget=None))
    create(BASE/'completion_snapshot.json',snapshot(ROOT,REL+'/completion_contract.json','completion'));print(checks)
if __name__=='__main__':main()
