"""Render FIT-only diagnostics, without refitting or resampling."""
import json
from .common import ROOT,BASE,read,sha,rel,unwrap
def put(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def main():
    out=BASE/'stage1_run';seal=unwrap(out/'completed.json')
    for n,h in seal['files'].items():
        if sha(out/n)!=h:raise ValueError('Changed stage1 output')
    s=unwrap(out/'SUMMARY.json');rows=['# L21 learned correction: FIT-only feasibility','',
        'Nested lexical-block cross-validation on24existingFITblocks/384nativevectorlabels. Historical frozenP/U is extra prior training. Eachouterfold trains256 andvalidates128labels. Zero newnative/suffix evaluations. This does NOT measure downstream readout utility.','',
        '| Method | Vector squared L2 | Coefficient squared L2 |','|---|---:|---:|']
    for k,v in s['outer_vector_L2'].items():rows.append(f"| {k} | {v:.9g} | {s['outer_coefficient_L2'][k]:.9g} |")
    rows+=['','C=P+U*learnedcoefficients; same_U is direct linear rank8refit; direct predictsfullD with independently selected linear/RBF model. Allmethods use same newtraininglabels. Models finalizedon384FITlabels, no futureTESTnative targets used.','',
        'No claim of learned readout improvement can be made before separate newholdout/native suffix measurements. Current64blockA TEST alreadyexposed, so upcoming stage2 uses new64blocks. Nominal futureblockintervals notindependentlyfamilyqualified.','',
        'Final chosen hyperparameters: '+json.dumps(s['selection'],ensure_ascii=False),
        'Independent completion status is recorded separately in completion_review.json.']
    (BASE/'REPORT_stage1.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    checks=dict(passed=True,outer_groups=3,blocks=24,conditions=384,native_forward=0,native_backward=0,stage2_launched=False)
    put(BASE/'completion_checks.json',checks)
    files=[p for p in out.rglob('*') if p.is_file()]+[BASE/n for n in ['REPORT_stage1.md','completion_checks.json','complete.py','pre_run_snapshot.json','pre_run_review.json','input_manifest.json','source_index.json','config.json']]
    put(BASE/'completion_manifest.json',dict(files={rel(p):sha(p) for p in files}))
    put(BASE/'completion_contract.json',dict(schema_version=1,id='A_L21_LEARNED_STAGE1_COMPLETION_20261103',kind='maintenance',primary_agent='/root',purpose='FIT-only feasibility and frozen384-label model evidence, no new fitting/inference.',
        tracked_files=[rel(BASE/n) for n in ['complete.py','REPORT_stage1.md','completion_checks.json']],tracked_trees=[],input_manifests=[rel(BASE/'completion_manifest.json')],
        gates=[dict(id='complete',category='accounting',evidence=rel(BASE/'completion_checks.json'),pointer='/passed',op='eq',expected=True,reason='Immutablecompletedstage1 outputs verified; actualresultscopeFITonly.')],circuits=[],structural_scope='Offline nestedvector/coefficientsanalysis only; no learnednative readout outcome.',entrypoint=None,schedule=None,budget=None))
    print(s['outer_vector_L2']);print(s['outer_coefficient_L2'])
if __name__=='__main__':main()
