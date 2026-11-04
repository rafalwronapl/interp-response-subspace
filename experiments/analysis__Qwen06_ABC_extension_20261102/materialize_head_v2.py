"""Preserve failed frozen v1; literal role-before-cohort compatibility fix."""
from pathlib import Path
import json
from analysis.C2_fresh_fit_holdout_20261102.worker import read,sha,ROOT
BASE=Path(__file__).parent
t=(BASE/'evaluate.py').read_text(encoding='utf-8');old="d['cohort']==cohort and d['split']==split and d['role']=='FIT'";new="d['role']=='FIT' and d['cohort']==cohort and d['split']==split";assert t.count(old)==1;t=t.replace(old,new);(BASE/'evaluate_v2.py').write_text(t,encoding='utf-8')
t=(BASE/'evaluation_worker.py').read_text(encoding='utf-8').replace('20261102.evaluate import','20261102.evaluate_v2 import');(BASE/'evaluation_worker_v2.py').write_text(t,encoding='utf-8')
t=(BASE/'prepare_evaluation.py').read_text(encoding='utf-8').replace("BASE/'evaluation_v1'","BASE/'evaluation_v2'").replace("'evaluate.py','score.py','evaluation_worker.py','prepare_evaluation.py'","'evaluate_v2.py','score.py','evaluation_worker_v2.py','prepare_evaluation_v2.py'").replace('20261102.evaluation_worker\'','20261102.evaluation_worker_v2\'');t=t.replace("'Qwen06_ABC_evaluation_v1'","'Qwen06_ABC_evaluation_v2'")
t=t.replace("BASE/'bundle_fixture_v1.json'", "BASE/'head_v2_schema_fixture.json',BASE/'materialize_head_v2.py',BASE/'bundle_fixture_v1.json'");(BASE/'prepare_evaluation_v2.py').write_text(t,encoding='utf-8')
cfg=read(BASE/'capture_base_v2/config.json');docs=read(ROOT/cfg['corpus'])['documents'];counts={i+'_'+s:len([d for d in docs if d['role']=='FIT' and d['cohort']==i and d['split']==s]) for i in ['r1','r2','r3'] for s in ['FIT_MAP','FIT_U']};assert set(counts.values())=={128};assert sum(d['role']=='HOLDOUT' and 'cohort' not in d for d in docs)==2048
proof=dict(status='PASS',model_calls=0,native_or_head_calls=0,scope='Actual metadata corpus:768FITdocs withcohort/split;2048holdoutdocs withoutFITcohort. Roleguardfirst selects6groups128each without touchingholdoutlabels.',counts=counts,v1_actual_failure_before_any_FIT_or_head=True,source_hashes={n:sha(BASE/n) for n in ['evaluate_v2.py','evaluation_worker_v2.py','prepare_evaluation_v2.py','materialize_head_v2.py']})
(BASE/'head_v2_schema_fixture.json').write_text(json.dumps(proof,indent=2)+'\n');print('v2actualcorpusrole-schemafixturePASS;0model/headcalls')
