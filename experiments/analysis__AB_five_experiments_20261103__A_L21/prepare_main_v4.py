"""Freeze executable main only after separately reviewed pretrained qualification."""
from pathlib import Path
import json,runpy,importlib.metadata,inspect,sys
from research_checks.gate import read,sha
from .worker_v4 import schedule
BASE=Path(__file__).parent;ROOT=BASE.parents[2];REL=BASE.relative_to(ROOT).as_posix()
def put(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
 draft='--draft' in sys.argv
 if not draft:
  completion=read(BASE/'pilot_run_v4/completed.json')['payload'];assert completion['ready']
  assert read(BASE/'pilot_completion_review.json')['verdict']=='approved'
 model=read(ROOT/'analysis/E041_execution/identity.json')['model_identity']['native']
 docs=read(BASE/'corpus.json')['documents'];put(BASE/'main_schedule.json',schedule(docs));put(BASE/'main_config.json',dict(model=model,output=REL+'/main_run'))
 import transformers.models.qwen3.modeling_qwen3 as qwen
 import transformers.masking_utils as mask
 put(BASE/'library_identity.json',dict(versions={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','safetensors']},sources={str(Path(inspect.getfile(m))):sha(Path(inspect.getfile(m))) for m in [qwen,mask]}))
 sources=[BASE/n for n in ['corpus.json','data_checks.json','freshness_inputs.json','main_schedule.json','main_config.json','main_fixtures.json','pipeline_fixture_v4.json','library_identity.json','PROTOCOL.txt','BATCH_AMENDMENT.txt','batch_fixture.json']]
 sources += [BASE/'status_fixture.json',ROOT/'runs/E036/development/predictors.npz',ROOT/'runs/E036/development/predictors.json',ROOT/'analysis/E041_execution/identity.json']
 sources += [p for p in (BASE/'pilot_run_v4').rglob('*') if p.is_file()]
 sources += [BASE/'pilot_snapshot_v4.json',BASE/'pilot_review_v4.json']
 if not draft:sources += [BASE/'pilot_completion_review.json',BASE/'pilot_completion_audit.json']
 sources += [ROOT/p for p in read(BASE/'freshness_inputs.json')['files']]
 sources += [p for p in (ROOT/'model').iterdir() if p.is_file()]
 put(BASE/'main_inputs.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(sources))}))
 helper=ROOT/'analysis/AB_confirmatory_execution_20261101/close_short_pilot_imports_v1.py'
 seeds=[REL+'/'+n for n in ['worker_safe.py','status_fixture.py','worker_v4.py','adapter_v4.py','math.py','main_fixtures.py','pipeline_fixture_v4.py','prepare_data.py','prepare_main_v4.py']]
 tracked=sorted(set(seeds)|set(runpy.run_path(str(helper))['closure'](seeds)))
 gates=[dict(id=k,category=k,evidence=REL+'/main_fixtures.json',pointer='/'+k,op='eq',expected=True,reason='Synthetic full-design fixtures; separately qualified native adapter') for k in ['hooks','data','pilot','accounting','restart']]
 gates.append(dict(id='trained_qualification',category='pilot',evidence=REL+'/pilot_run_v4/completed.json',pointer='/payload/ready',op='eq',expected=True,reason='35-logical-evaluation batched full/suffix parity and sensitivity'))
 gates.append(dict(id='advisory_status',category='restart',evidence=REL+'/status_fixture.json',pointer='/ok',op='eq',expected=True,reason='Windows telemetry lock cannot interrupt measurements; immutable errors still propagate'))
 gates.append(dict(id='whole_worker_fixture',category='restart',evidence=REL+'/pipeline_fixture_v4.json',pointer='/ok',op='eq',expected=True,reason='Actual worker synthetic end-to-end, forecast ordering, replay zero calls, corruption fail-closed'))
 put(BASE/'main_contract.json',dict(schema_version=1,id='A_L21_fresh64_and_ranks_20261103',kind='experiment',primary_agent='/root/a_l21_five_campaign',purpose='Frozen historical B5/F1 replication on64newblocks plus separate3FIT rank/control study',tracked_files=tracked,tracked_trees=[],input_manifests=[REL+'/main_inputs.json'],gates=gates,circuits=[],structural_scope='Actual L19neuron intervention/L20restore, targetL21, dynamicL22-27; no reduced circuit.',entrypoint=dict(module='analysis.AB_five_experiments_20261103.A_L21.worker_safe',args=[]),schedule=REL+'/main_schedule.json',budget=dict(forward=66656,backward=0)))
 print('Draft main requires completed pilot and reseal' if draft else 'Main ready for independent review; no modelcalls')
if __name__=='__main__':main()

