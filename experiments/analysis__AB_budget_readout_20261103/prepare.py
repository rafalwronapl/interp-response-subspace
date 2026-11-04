"""Bind current native TEST worker, all actual inputs and historical qualification."""
from pathlib import Path
import json,runpy,inspect,importlib.metadata
from research_checks.gate import read,sha
from .worker import schedule
BASE=Path(__file__).parent;ROOT=BASE.parents[1];LEARNED=ROOT/'analysis/A_L21_learned_20261103';OLD=ROOT/'analysis/AB_five_experiments_20261103/A_L21';PREVIOUS=LEARNED/'stage2';REL=BASE.relative_to(ROOT).as_posix()
def put(n,v):(BASE/n).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
    assert read(PREVIOUS/'completion_review.json')['verdict']=='approved'
    assert read(ROOT/'analysis/AB_fast_followups_20261103/A_budget/review/completion_review.json')['verdict']=='approved'
    assert read(BASE/'fixture.json')['worker_sha256']==sha(BASE/'worker.py')
    assert read(BASE/'fixture.json')['synthetic_counter']['max_batch']==16
    index=read(LEARNED/'source_index.json');selected=sorted([e for e in index['cells'] if e['document']['split']=='FIT0'],key=lambda e:e['id'])[:2]
    quals=[dict(id=e['id'],document=e['document'],natural=index['natural'][e['document']['id']],prefix=e['prefix'],target=e['target']) for e in selected]
    put('qualification.json',dict(contexts=quals,selection='First2 lexicographic historicalFIT0 CONDITIONIDs; no newTESTlabels'))
    docs=read(BASE/'corpus.json')['documents'];ops=schedule(docs,quals);assert len(ops['operations'])==18726;put('operations.json',ops)
    model=read(PREVIOUS/'config.json')['model'];put('config.json',dict(model=model,checkpoint='model',prior='runs/E036/development/predictors.npz',models=REL+'/MODELS.json',dimension=1024,test_blocks=64,maximum_forward=18726,output=REL+'/run',selection='FIT0seed17',primary='C_32-Direct_32'))
    import transformers.models.qwen3.modeling_qwen3 as qwen
    import transformers.masking_utils as masks
    put('library_identity.json',dict(versions={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','safetensors']},sources={str(Path(inspect.getfile(m))):sha(Path(inspect.getfile(m))) for m in [qwen,masks]}))
    paths=[BASE/n for n in ['corpus.json','data_checks.json','freshness_inputs.json','qualification.json','operations.json','config.json','library_identity.json','fixture.json','MODELS.json','model_checks.json','model_replay_checks.json','PROTOCOL.txt']]
    paths += list((BASE/'materialized_fits').glob('*.json'))
    paths += [PREVIOUS/n for n in ['input_manifest.json','pre_run_snapshot.json','pre_run_review.json','completion_snapshot.json','completion_review.json','run/completed.json']]
    paths += [OLD/n for n in ['pilot_run_v4/completed.json','pilot_completion_review.json','pilot_completion_audit.json','pilot_snapshot_v4.json','main_run/completed.json','completion_review.json']]
    paths += [ROOT/'plans/A_L21_LEARNED_20261103/PLAN.txt',ROOT/'plans/A_L21_LEARNED_20261103/review/plan_review_v1.json']
    paths += [p for p in (ROOT/'model').iterdir() if p.is_file()]
    paths += [ROOT/p for p in read(BASE/'freshness_inputs.json')['files']]
    paths += [ROOT/p for p in read(ROOT/'analysis/AB_fast_followups_20261103/A_budget/input_manifest.json')['files']]
    budget=ROOT/'analysis/AB_fast_followups_20261103/A_budget'
    paths += [budget/n for n in ['input_manifest.json','review/completion_review.json','completion_snapshot_v3.json','run/completed.json']]
    paths += [budget/'run/subsets'/f'FIT0_s17_n{n}.json' for n in [32,64,128,256]]
    paths += [LEARNED/'stage1_run/MODELS.json',LEARNED/'stage1_run/completed.json']
    for q in quals:
        for k in ['natural','prefix','target']:paths += [ROOT/q[k]['receipt'],ROOT/q[k]['cache']]
    seeds=[REL+'/'+n for n in ['worker.py','data.py','fixture.py','prepare.py','models.py','check_models.py']]
    helper=ROOT/'analysis/AB_confirmatory_execution_20261101/close_short_pilot_imports_v1.py'
    tracked=sorted(set(seeds)|set(runpy.run_path(str(helper))['closure'](seeds)))
    paths += [ROOT/p for p in tracked]
    put('input_manifest.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}))
    gates=[dict(id=k,category=k,evidence=REL+'/fixture.json',pointer='/'+k,op='eq',expected=True,reason='Actual16arm worker synthetic endtoend globalfreeze/restart/identity/alias/cost checks') for k in ['hooks','data','pilot','accounting','restart']]
    gates += [dict(id='fresh_data',category='data',evidence=REL+'/data_checks.json',pointer='/ok',op='eq',expected=True,reason='64recordedfreshpairs/prompts including previouslearnedTEST exclusion'),dict(id='frozen_models',category='data',evidence=REL+'/model_checks.json',pointer='/payload/passed',op='eq',expected=True,reason='Frozen historicchoices,necessary6jointmaterializations,12reproduced savedlosses;384packedweights reused')]
    put('contract.json',dict(schema_version=1,id='A_BUDGET_FRESH64_READOUT_20261103',kind='experiment',primary_agent='/root',purpose='Fixedsequence budgetcurve on64newblocks;C32-Direct32 primary,allforecasts beforeTESTtarget,16armnative suffix',tracked_files=tracked,tracked_trees=[],input_manifests=[REL+'/input_manifest.json'],gates=gates,circuits=[],structural_scope='Unchangedqualified Adapter_v4 nativeL19neuron1317/L20restore1423,MLP21patch and dynamicL22-27suffix.2FITcontexts:3fullrefs+16paddedtails each;38qualification logicalcalls beforeTEST',entrypoint=dict(module='analysis.AB_budget_readout_20261103.worker',args=[]),schedule=REL+'/operations.json',budget=dict(forward=18726,backward=0)))
    print(dict(ready_for_independent_review=True,forward=18726,backward=0,physical_maximum=3336,batch=16))
if __name__=='__main__':main()
