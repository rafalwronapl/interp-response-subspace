"""Offline completion/report snapshot; does not load pretrained model."""
from pathlib import Path
import json
from research_checks.gate import read,sha
from analysis.A_L21_learned_20261103.common import unwrap
BASE=Path(__file__).parent;ROOT=BASE.parents[1];REL=BASE.relative_to(ROOT).as_posix()
def put(n,v):(BASE/n).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
 out=BASE/'run';done=unwrap(out/'completed.json');summary=unwrap(out/'summary.json');models=unwrap(out/'MODELS.json')
 assert done['complete'] and done['conditions']==1024 and done['physical_forward_calls']==4794 and done['attempts']==30444
 selected=[{name:{'selected':b['scores'][name]['selected'],'param':b['models'][name]['param']} for name in b['models']} for b in models['bundles']]
 result=dict(summary=summary,selected_parameters=selected,completed=done,review_status='pending independent completion review')
 put('analysis_summary.json',result)
 lines=['Matched historical-prior comparison:6freshFIT32/commonfreshTEST64','2026-11-04. Measurements complete; independent completion review pending.',
 'Scope: Qwen3-0.6B-Base L21, exact historical E041 F1/is-are regime, branches B/C,doses.5/1. Twelve new FIT lexical blocks,192fullvectorlabels;64new TEST lexical blocks,1024conditions. One common TEST for all six fits.',
 'Primary C32−Direct+P32 is/are margin MSE (negative favors C): '+json.dumps(summary['primary']),
 'Interval resamples TEST lexical blocks conditional on all six fixed FIT cohorts. It excludes FIT sampling uncertainty; six FIT cohorts support descriptive variation, not a population stability claim.',
 'All contrasts including per-FIT values: '+json.dumps(summary['contrasts']),
 'Per-arm MSE: '+json.dumps({k:v['MSE'] for k,v in summary['metrics'].items()}),
 'Selected parameters per FIT: '+json.dumps(selected),
 'Every method independently selects from identical8linear/RBF candidates using2fold whole-block vector-loss CV inside32labels. Two blocks give noisy selection; parameter differences remain part of comparison.',
 'Linear residual C is ridge with historical coefficient prior W, up to actual FP32 rounding of P. Direct+P=P+F(D−P) supplies the missing full-vector prior comparator; it is not the algebraically identical same-U-with-P method. No algorithm novelty claim.',
 'Resources: C/Direct+P use historical W,U,scales; same-U uses U,scales; Direct uses scales. Historical E036 resources512labels/640network-CLT/512JVP are additional inherited resources, not new-run calls or free information. New labels matched within each cohort.',
 'Physical4794forward, logical30444arm-equivalents,0backward; model_load_attempts='+str(done['model_load_attempts'])+'. TEST25forecastarms+native/zeroMLPdelta=27targets split16/11. Full/suffix parity and sensitivity receipts in run/qualification. No claimed total-resource savings.',
 'TEST raw vector forecasts and frozen models published globally before first TEST intervention target. All76 unordered noun pairs excluded against815 inventoried historical sources, including previous3FIT and priorTEST; freshness claim is limited to inventory.',
 'Zero arm is natural MLP21 output inserted at the intervened residual (zero MLPdelta), not original prompt natural margin. Actual original natural prompt margin stored separately. Native reference uses common suffix numerical path; qualification bounds full/suffix difference.',
 'Evidence: run/completed.json, run/summary.json, run/FIT_vectors.json, run/MODELS.json, run/TEST_forecasts.json, run/rows/, run/ledger.jsonl, input_manifest.json, data_checks.json, fixture.json, pre_run_snapshot.json and review/pre_run_review.json.',
 'Independent completion status: pending; do not present as approved until actual reviewer receipt.']
 (BASE/'REPORT.txt').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
 paths={ROOT/p for p in read(BASE/'input_manifest.json')['files']};paths.update(p for p in out.rglob('*') if p.is_file());paths.update(BASE/n for n in ['REPORT.txt','analysis_summary.json','pre_run_snapshot.json','contract.json','review/pre_run_review.json'])
 put('completion_manifest.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}))
 put('completion_contract.json',dict(schema_version=1,id='A_PRIOR_MATCHED_COMPLETION_20261104',kind='maintenance',primary_agent='/root/a_prior_experiment',purpose='Offline actualworker integrity, prospective matched-prior comparison and bounded claims completion review',tracked_files=[],tracked_trees=[],input_manifests=[REL+'/completion_manifest.json'],gates=[],circuits=[],structural_scope='No new measurement in completion; reviewed fullnative/suffix experiment already executed',entrypoint=None,schedule=None,budget=None))
 print(dict(completed=True,physical=done['physical_forward_calls'],primary=summary['primary'],completion_review_required=True))
if __name__=='__main__':main()
