import json
from pathlib import Path
import numpy as np
from research_checks.gate import read,digest
from analysis.A_L21_learned_20261103.common import unwrap
R=Path(__file__).resolve().parents[3];B=R/'analysis/A_transfer_next_20261103';O=B/'run';ats={json.loads(l)['record']['operation']:json.loads(l)['record'] for l in (O/'ledger.jsonl').read_text().splitlines() if json.loads(l)['record']['event']=='attempt'}
def op(n):
 c=unwrap(O/'cache'/(ats[n]['request']+'.json'));assert digest(c['request'])==ats[n]['request'];return c
qs=[]
for q in range(2):
 full=[op(f'qual{q}_full{i}') for i in range(3)];tail=op(f'qual{q}_tail0');ys=tail['request']['intervention']['targets'];assert len(ys)==3
 for i,f in enumerate(full):assert f['request']['intervention']['target']==ys[i]
 np.testing.assert_array_equal(np.asarray(np.asarray(ys[0],np.float32)+.01*np.linspace(.5,1.5,len(ys[0])),np.float32),ys[2])
 expected=np.asarray([f['result'] for f in full]);actual=np.asarray(tail['result']);err=float(np.max(np.abs(expected-actual)));sens=np.abs(actual[0]-actual[2]);assert err<=1e-4 and np.min(sens)>1e-8
 recorded=unwrap(O/'qualification'/f'{q}.json');assert recorded['passed'] and recorded['maxabs']==err;np.testing.assert_array_equal(recorded['fixed_sensitivity'],sens);qs.append(dict(maxabs=err,sensitivity=sens.tolist()))
raw=unwrap(O/'summary.json');corrected=read(B/'analysis_summary.json')
for head in raw['readouts']:
 r=dict(raw['readouts'][head])
 for old,new in [('natural_margin_range','zero_MLPdelta_baseline_margin_range'),('natural_margin_sign_counts','zero_MLPdelta_baseline_sign_counts'),('native_effect_MSE','finite_MLPdelta_effect_MSE_conditional_upstream'),('native_effect_sign_counts','finite_MLPdelta_effect_sign_counts_conditional_upstream')]:r[new]=r.pop(old)
 assert r==corrected['readouts'][head]
report=(B/'REPORT.txt').read_text();assert 'NOT original natural prompt/readout' in report and 'Original has/have natural-prompt margins were not measured' in report
out=dict(passed=True,qualifications=qs,semantic_amendment_only_renames_four_fields=True,pretrained_calls=0)
(B/'review/independent_controls_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
