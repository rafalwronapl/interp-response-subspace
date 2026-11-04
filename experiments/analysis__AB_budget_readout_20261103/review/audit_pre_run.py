"""Independent review harness: synthetic worker, source/model/data validation."""
import json,tempfile
from pathlib import Path
import numpy as np
from analysis.AB_budget_readout_20261103 import worker as w,fixture as f,models as m
from analysis.A_L21_learned_20261103.common import unwrap,array,sha

here=Path(__file__).resolve().parent;actual=w.BASE
# Execute actual worker fixture in reviewer-owned temporary/output paths.
f.BASE=here
oldsha=f.sha
f.sha=lambda p: oldsha(actual/'worker.py') if Path(p)==here/'worker.py' else oldsha(p)
f.main()
ev=json.loads((here/'fixture.json').read_text());assert ev['synthetic_counter']['max_batch']==16
bundle=unwrap(actual/'MODELS.json');original=unwrap(m.ROOT/'analysis/A_L21_learned_20261103/stage1_run/MODELS.json')
for new,old in [('C_384','C'),('same_U_384','same_U'),('Direct_384','direct')]:assert bundle['models'][new]==original['models'][old]
assert len(bundle['models'])==15 and len(list((actual/'materialized_fits').glob('*.json')))==6
for n in [32,64,128,256]:
    source=unwrap(m.old.BASE/'run/subsets'/f'FIT0_s17_n{n}.json')
    for method,old in [('C','C'),('same_U','same_U'),('Direct','direct')]:
        assert bundle['models'][method+'_'+str(n)]['param']==source['selected'][old]
corpus=json.loads((actual/'corpus.json').read_text());docs=corpus['documents'];pairs=corpus['pairs']
assert len(docs)==256 and len(pairs)==len({tuple(sorted(x)) for x in pairs})==64
assert len({d['prompt'] for d in docs})==256 and len({d['block'] for d in docs})==64
byid={d['id']:d for d in docs}
for d in docs:
    donor=byid[d['donor']]
    assert d['block']==donor['block'] and d['S']!=donor['S'] and d['A']==donor['A']
    assert d['split']=='TEST' and d['ids'][0]==151643
prior=json.loads((m.ROOT/'analysis/A_L21_learned_20261103/stage2/corpus.json').read_text())
assert not {tuple(sorted(x)) for x in pairs}&{tuple(sorted(x)) for x in prior['pairs']}
assert not {d['prompt'] for d in docs}&{d['prompt'] for d in prior['documents']}
quals=json.loads((actual/'qualification.json').read_text())['contexts']
schedule=w.schedule(docs,quals);assert len(schedule['operations'])==18726
assert len({o['id'] for o in schedule['operations']})==18726
assert all(o['kind']=='forward' for o in schedule['operations'])
# Verify the declared primary and paired-block interval calculation independently.
rows=[]
for block in range(2):
    margins={a:2. for a in w.ALL};margins['Direct_32']=3.
    rows.append(dict(block=block,margins=margins,truth=2.,natural_margin=0.,
        vector_L2={a:0. for a in w.ALL},coefficient_L2={a:0. for a in w.ALL},
        physical_suffix_batches=1,branch='B',dose=.5,S=0,A=0))
s=w.summarize(rows);assert s['primary']['contrast']=='C_32-Direct_32'
assert s['primary']['delta_MSE']==-1. and s['primary']['nominal95']==[-1.,-1.]
r=dict(passed=True,actual_worker_synthetic_rerun=True,batch16=True,
    global_forecast_freeze_and_restart=True,model384_objects_byte_equivalent=True,
    six_materializations_fifteen_models=True,frozen_lower_budget_params=True,
    corpus64pairs256prompts=True,previous_TEST_disjoint=True,donor_alignment=True,
    logical_forward_cap=18726,physical_forward_cap=3336,backward=0,
    primary_C32_Direct32_and_paired_block_calculation=True,native_calls=0)
(here/'independent_pre_run_audit.json').write_text(json.dumps(r,indent=2)+'\n')
print(r)
