"""Seal stage2 only after actual stage1 completion and independent acceptance."""
from pathlib import Path
import json,runpy,inspect,importlib.metadata
from research_checks.gate import read,sha
from .worker import schedule
BASE=Path(__file__).parent;PARENT=BASE.parent;ROOT=BASE.parents[2];OLD=ROOT/'analysis/AB_five_experiments_20261103/A_L21';REL=BASE.relative_to(ROOT).as_posix()
def put(n,v):(BASE/n).write_text(json.dumps(v,indent=2)+'\n')
def main():
 review=read(PARENT/'completion_review.json');assert review['verdict']=='approved' and review['independent']
 assert (PARENT/'stage1_run/completed.json').exists() and (PARENT/'stage1_run/MODELS.json').exists()
 index=read(PARENT/'source_index.json');selected=sorted([e for e in index['cells'] if e['document']['split']=='FIT0'],key=lambda e:e['id'])[:2]
 quals=[dict(id=e['id'],document=e['document'],natural=index['natural'][e['document']['id']],prefix=e['prefix'],target=e['target']) for e in selected]
 put('qualification.json',dict(contexts=quals,selection='First2 lexicographic FIT0 CONDITION IDs from fixed stage1 source_index; existing native caches only'))
 docs=read(BASE/'corpus.json')['documents'];ops=schedule(docs,quals);assert len(ops['operations'])==8460;put('operations.json',ops)
 model=read(OLD/'main_config.json')['model'];put('config.json',dict(model=model,checkpoint='model',prior=read(PARENT/'config.json')['prior'],models=(PARENT/'stage1_run/MODELS.json').relative_to(ROOT).as_posix(),dimension=1024,test_blocks=64,maximum_forward=8460,output=REL+'/run',training_new_labels=384,prior_training_extra=True))
 import transformers.models.qwen3.modeling_qwen3 as qwen
 import transformers.masking_utils as masks
 put('library_identity.json',dict(versions={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','safetensors']},sources={str(Path(inspect.getfile(m))):sha(Path(inspect.getfile(m))) for m in [qwen,masks]}))
 paths=[BASE/n for n in ['corpus.json','data_checks.json','freshness_inputs.json','qualification.json','operations.json','config.json','library_identity.json','fixture.json']]
 paths += [PARENT/n for n in ['source_index.json','common.py','models.py','stage1_run/MODELS.json','stage1_run/completed.json','completion_review.json','completion_snapshot.json','completion_checks.json']]
 paths += [OLD/n for n in ['pilot_run_v4/completed.json','pilot_completion_review.json','pilot_completion_audit.json','pilot_snapshot_v4.json','main_run/completed.json','completion_review.json']]
 paths += [ROOT/'plans/A_L21_LEARNED_20261103/PLAN.txt',ROOT/'plans/A_L21_LEARNED_20261103/review/plan_review_v1.json',ROOT/read(PARENT/'config.json')['prior']]
 paths += [p for p in (ROOT/'model').iterdir() if p.is_file()]
 paths += [ROOT/p for p in read(BASE/'freshness_inputs.json')['files']]
 for q in quals:
  for k in ['natural','prefix','target']:
   paths += [ROOT/q[k]['receipt'],ROOT/q[k]['cache']]
 seeds=[REL+'/'+n for n in ['worker.py','data.py','fixture.py','prepare.py']]+[(PARENT/n).relative_to(ROOT).as_posix() for n in ['models.py','common.py']]
 helper=ROOT/'analysis/AB_confirmatory_execution_20261101/close_short_pilot_imports_v1.py';tracked=sorted(set(seeds)|set(runpy.run_path(str(helper))['closure'](seeds)))
 paths += [ROOT/p for p in tracked]
 put('input_manifest.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}))
 gates=[dict(id=k,category=k,evidence=REL+'/fixture.json',pointer='/'+k,op='eq',expected=True,reason='Actual worker synthetic end-to-end, frozenforecast ordering, exactaliases, completedreplay0calls, corruptionfailclosed') for k in ['hooks','data','pilot','accounting','restart']]
 gates += [dict(id='fresh_data',category='data',evidence=REL+'/data_checks.json',pointer='/ok',op='eq',expected=True,reason='64 fresh unordered pairs and prompts; all88priorcampaign FIT/TESTpairs excluded')]
 put('contract.json',dict(schema_version=1,id='A_L21_LEARNED_FRESH64_STAGE2_20261103',kind='experiment',primary_agent='/root/a_l21_five_campaign',purpose='Prospective non-oracle learned correction versus historicalP/sameU/Direct384/mean andoracle reference; allforecasts frozen before TESTtarget,64freshblocks',tracked_files=tracked,tracked_trees=[],input_manifests=[REL+'/input_manifest.json'],gates=gates,circuits=[],structural_scope='Unchanged qualified nativeL19neuron1317/L20restore1423 intervention, targetL21 and full dynamicL22-27 suffix;12FIT full/suffixparitycalls beforeTEST',entrypoint=dict(module='analysis.A_L21_learned_20261103.stage2.worker',args=[]),schedule=REL+'/operations.json',budget=dict(forward=8460,backward=0)))
 print(dict(ready_for_independent_pre_run=True,maximum_forward=8460,backward=0,new_FIT_acquisition=0,qualification_ids=[q['id'] for q in quals]))
if __name__=='__main__':main()
