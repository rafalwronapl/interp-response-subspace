"""Report only actual sealed results; never starts a measurement."""
from pathlib import Path
import json
from analysis.A_L21_learned_20261103.common import read,unwrap,sha
from analysis.AB_budget_readout_20261103.worker import outputs
BASE=Path(__file__).parent
ROOT=BASE.parents[1];REL=BASE.relative_to(ROOT).as_posix()
def main():
 done=unwrap(BASE/'run/completed.json');assert done['complete'] and done['files']==outputs(BASE/'run')
 summary=unwrap(BASE/'run/summary.json');models=unwrap(BASE/'run/MODELS.json');text=['A new grammatical readout and three independent FIT32 cohorts, 2026-11-03','',
 'Completed96 new vector labels across3 disjoint2-block FIT cohorts. Each32-label fit has only2 independent lexical blocks; every CV fold trains on1block. Historical E036 P/U/scales remain extra prior information.',
 'TEST reuses the completed64-block F1 cohort posthoc, selected after the favorable original C32 result. This is a new have-minus-has grammatical next-token readout on the same contexts/model/operator, not new-context/model transfer or independent confirmation.',
 'All9 learned forecasts and P were frozen globally before new TEST readout calls. Both readouts use the same12-vector native suffix batches. Vector-label selection used no readout targets. All outcomes retained.','']
 for head,r in summary['readouts'].items():
  text.extend([head, 'MSE by arm: '+json.dumps({a:m['MSE'] for a,m in r['metrics'].items()},sort_keys=True),
   'Mean across3 fits C32-Direct32 deltaMSE='+str(r['mean_C_Direct_delta'])+'; block bootstrap nominal95='+str(r['nominal95']),
   'PerFIT: '+json.dumps(r['per_fit_C_Direct'],sort_keys=True),'Finite MLP-delta readout effect at fixed intervened upstream, MSE='+str(r['native_effect_MSE'])+'; signs='+str(r['native_effect_sign_counts']),
   'Zero MLP-delta baseline margin range='+str(r['natural_margin_range'])+'; signs='+str(r['natural_margin_sign_counts']),''])
 corrected=json.loads(json.dumps(summary))
 for r in corrected['readouts'].values():
  for old,new in [('natural_margin_range','zero_MLPdelta_baseline_margin_range'),('natural_margin_sign_counts','zero_MLPdelta_baseline_sign_counts'),('native_effect_MSE','finite_MLPdelta_effect_MSE_conditional_upstream'),('native_effect_sign_counts','finite_MLPdelta_effect_sign_counts_conditional_upstream')]:r[new]=r.pop(old)
 corrected['semantic_amendment']='zero restores naturalMLP21y at last token with intervenedprefix;not originalpromptnaturalreadout. Rawsummary preserved. Primaryerrorsunchanged.'
 (BASE/'analysis_summary.json').write_text(json.dumps(corrected,indent=2)+'\n')
 text.extend(['Primary is has_have mean C32-Direct32. Negative delta favors C; positive favors Direct. is_are and auxiliary comparisons are descriptive. Bootstrap1999 draws seed3111038 resamples64 blocks conditional on these3 fits; it does not estimate training-population uncertainty.',
 'Post-run semantic correction: frozen worker field names natural_margin_range, natural_margin_sign_counts and native_effect are misleading. The arm zero patches natural MLP21 y into the INTERVENED upstream residual; it is zero predicted MLP-delta, NOT original natural prompt/readout or zero whole intervention. Its effect statistics hold the upstream intervention fixed. The old raw summary is preserved; this report uses the corrected meaning. Original has/have natural-prompt margins were not measured. This does not alter learned/native readout errors or primary C-Direct comparisons.',
 'No claim of equal-total-information superiority, interpretability of the learned predictor, worldwide novelty, or general model understanding. Native target-free prefixes remain necessary at prediction time.',
 'Accounting: '+json.dumps({k:v for k,v in done.items() if k!='files'},sort_keys=True),
 'FIT selection: '+json.dumps([{m:b['scores'][m]['selected'] for m in b['scores']} for b in models['bundles']]),
 'Evidence: run/summary.json, run/MODELS.json, run/TEST_forecasts.json,1024immutable rows, raw cache/bundles/ledger, run/completed.json, qualification/.',
 'Pre-run independent approval: review/pre_run_review.json. Completion review is required separately; this report does not substitute for approval.'])
 (BASE/'REPORT.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
 snapshot=read(BASE/'pre_run_snapshot.json')['snapshot_sha256'];receipts=[p for p in (ROOT/'operations').glob('research_launch_*.json') if read(p).get('snapshot_sha256')==snapshot]
 assert len(receipts)==1 and read(receipts[0])['returncode']==0,'Need actual canonical launcher exit0 receipt'
 paths={ROOT/p for p in read(BASE/'input_manifest.json')['files']};paths.update(p for p in (BASE/'run').rglob('*') if p.is_file() and p.name!='STATUS.tmp');paths.update(receipts)
 paths.update(BASE/n for n in ['PROTOCOL.txt','REPORT.txt','analysis_summary.json','complete.py','pre_run_snapshot.json','review/pre_run_review.json','contract.json','input_manifest.json','canonical_verify.log'])
 manifest=dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)})
 (BASE/'completion_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 contract=dict(schema_version=1,id='A_NEW_READOUT_3FIT32_COMPLETION_20261103',kind='maintenance',primary_agent='/root',purpose='Review sealed new grammatical readout and3freshFIT32 on oldTEST,includingnegativeoutcomes',tracked_files=[REL+'/complete.py',REL+'/REPORT.txt'],tracked_trees=[],input_manifests=[REL+'/completion_manifest.json'],gates=[dict(id='completed',category='accounting',evidence=REL+'/run/completed.json',pointer='/payload/complete',op='eq',expected=True,reason='Immutable output seal and fullphysical/logicalledger reconciliation')],circuits=[],structural_scope='Exact established native boundary/suffix;technicalparityandfixedsensitivitybothheads passed;completiononly',entrypoint=None,schedule=None,budget=None)
 (BASE/'completion_contract.json').write_text(json.dumps(contract,indent=2)+'\n');print(dict(report=True,completed_sha256=sha(BASE/'run/completed.json'),completion_files=len(paths)))
if __name__=='__main__':main()
