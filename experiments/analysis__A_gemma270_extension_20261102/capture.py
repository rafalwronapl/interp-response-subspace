"""A270 fresh acquisition; exact qualified Native, no FIT or arm changes."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import argparse,json,hashlib,shutil,runpy
from pathlib import Path
from analysis.C2_fresh_fit_holdout_20261102.worker import read,sha,receipt,plain,put
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent
CORPUS_SHA='0710f9ee1d623997fff228fed43952e1b9e44774f6f74cd5030330aca07d5676'

def qualification(cfg):
 from research_checks.gate import digest
 s=read(ROOT/cfg['qualification_snapshot']);r=read(ROOT/cfg['qualification_review']);assert digest(s['payload'])==s['snapshot_sha256']=='72c865d6e7136735c1010b26aae45556259b9e52d7eebd558ede5cf154b8bd11';assert r['snapshot_sha256']==s['snapshot_sha256'] and r['verdict']=='approved' and r['independent'] is True
 for p in [ROOT/cfg['model_identity'],ROOT/cfg['qualified_controls'],ROOT/'ab_confirmatory_20261101/native.py']:
  assert sha(p)==s['payload']['files'][p.relative_to(ROOT).as_posix()]
 controls=receipt(ROOT/cfg['qualified_controls']);assert len(controls)==4 and all(c['passed'] and c['parity']<=1e-4 and c['branch']<=1e-4 and c['sensitivity']>=1e-6 for c in controls)

def bind_fixed_pairs(cfg):
 old=ROOT/'analysis/C3_noD_more_models_20261102/gemma270_evaluation_v1'
 cfg['fit_snapshot']=(old/'completion_snapshot_v1.json').relative_to(ROOT).as_posix();cfg['fit_review']=(old/'completion_review_resume_v1.json').relative_to(ROOT).as_posix()
 cfg['fixed_pairs']=[dict(fit=i,P=(old/'run/fits'/('P_'+i+'.json')).relative_to(ROOT).as_posix(),U=(old/'run/fits'/('U_'+i+'.json')).relative_to(ROOT).as_posix()) for i in ['r1','r2','r3']]
 for pair in cfg['fixed_pairs']:
  for key in ['P','U']:pair[key+'_sha256']=sha(ROOT/pair[key])
 return cfg

def restore_cached_prefix(runtime,docs,tokens,cfg):
 for d in docs:
  op='capture_'+d['id']
  if op in runtime.attempts:
   def forbidden():raise AssertionError('Consumed callback must never be invoked on resume')
   runtime.compute(op,dict(document=d,tokens=tokens[d['id']]),dict(mode='natural'),dict(answer_ids=cfg['answer_ids'],orientation='is-minus-are'),forbidden)

def prepare():
 from transformers import AutoTokenizer
 stage=BASE/'capture_v1';stage.mkdir();old=ROOT/'analysis/C3_noD_more_models_20261102/gemma270_capture';oldcfg=read(old/'config.json');ident=read(ROOT/oldcfg['model_identity'])['A'];tok=AutoTokenizer.from_pretrained(ident['folder'],local_files_only=True);corpus=BASE/'corpus.json';assert sha(corpus)==CORPUS_SHA;docs=read(corpus)['documents'];assert len(docs)==2048
 answers=[tok.encode(x,add_special_tokens=False) for x in [' is',' are']];assert all(len(x)==1 for x in answers);answers=[x[0] for x in answers];assert answers==oldcfg['assets']['answer_ids'];tokens={}
 for d in docs:
  ids=tok(d['text'])['input_ids'];assert 1<len(ids)<=256 and (tok.unk_token_id is None or tok.unk_token_id not in ids);tokens[d['id']]=dict(ids=ids,position=len(ids)-1,donor=d['donor'],answer_ids=answers)
 rel=stage.relative_to(ROOT).as_posix();cfg=dict(model='gemma3-270m',corpus=corpus.relative_to(ROOT).as_posix(),corpus_sha256=CORPUS_SHA,tokens=rel+'/tokens.json',output=rel+'/run',schedule=rel+'/operations.json',model_identity=oldcfg['model_identity'],answer_ids=answers,assets=oldcfg['assets'],asset_path=oldcfg['asset_path'],qualification_snapshot=(old/'completion_snapshot_v1.json').relative_to(ROOT).as_posix(),qualification_review=(old/'completion_review_resume_v1.json').relative_to(ROOT).as_posix(),qualified_controls=(old/'run/controls.json').relative_to(ROOT).as_posix(),threads=1,minimum_output_disk_free_bytes=1073741824,estimated_output_reserve_bytes=268435456,layer=17,scope='New256blocksT2T5,3fixedC3P/U0 pairs; no new FIT/technicalcontrol repeats')
 cfg=bind_fixed_pairs(cfg);qualification(cfg);put(stage/'tokens.json',tokens);put(stage/'config.json',cfg);put(stage/'operations.json',dict(operations=[dict(id='capture_'+d['id'],kind='forward') for d in docs]));put(stage/'evidence.json',dict(fresh_corpus=True,qualified_native_reused=True,capture=2048,backward=0,LLM_loads=1,output_drive='C',no_new_FIT=True))
 files=[corpus,BASE/'freshness_audit_v2.json',BASE/'capture_paths_index_v1.json',ROOT/'analysis/AB_confirmatory_execution_20261101/corpus/observed_words_v2.json',ROOT/cfg['model_identity'],ROOT/cfg['qualification_snapshot'],ROOT/cfg['qualification_review'],ROOT/cfg['qualified_controls'],stage/'tokens.json',stage/'config.json',stage/'operations.json',ROOT/cfg['asset_path']]
 files += [ROOT/cfg['fit_snapshot'],ROOT/cfg['fit_review']]+[ROOT[p[k]] for p in cfg['fixed_pairs'] for k in ['P','U']];put(stage/'input_manifest.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in files}));helper='analysis/AB_confirmatory_execution_20261101/close_short_pilot_imports_v1.py';seeds=[BASE.relative_to(ROOT).as_posix()+'/'+n for n in ['capture.py','prepare_corpus.py','PROTOCOL.md','analyze.py','restart_fixture.py','restart_fixture_v1.json']]+[helper];tracked=sorted(set(seeds)|set(runpy.run_path(str(ROOT/helper))['closure'](seeds)))
 put(stage/'contract.json',dict(schema_version=1,id='A_G270_EXTENSION_CAPTURE_V1',kind='experiment',primary_agent='/root/sol_parallel_acceptance',purpose='Fresh256lexicalblocks,T2T5;primarythreefixedFIT-averageinside-P onT5;outside descriptive. Captureonly, C-only storage, no refit.',tracked_files=tracked,tracked_trees=[],input_manifests=[rel+'/input_manifest.json'],gates=[dict(id=k,category=k,evidence=rel+'/evidence.json',pointer='/fresh_corpus',op='eq',expected=True,reason='Actual global exposure/lexicalaudit; exactpreviouslyqualified Native/weights/tokensreadout; frozen analysis, no new data-dependent choices.') for k in ['hooks','data','pilot','accounting','restart']],circuits=[],structural_scope='Full-model normalizedlastMLP capture; exact reusedGemmaL17nativeadapter and C3parity/controlproof. New data and fixedmaps; no transfer to oldB-L15TCoperator claim.',entrypoint=dict(module='analysis.A_gemma270_extension_20261102.capture',args=['run','--config',rel+'/config.json']),schedule=rel+'/operations.json',budget=dict(forward=2048,backward=0)))
 print('ACTUAL capture packet prepared:2048F/0B/1load')

def run(config):
 from research_checks.gate import digest
 from research_checks.persistence import immutable
 from ab_confirmatory_20261101.runtime import Runtime
 cfg=read(ROOT/config);qualification(cfg);assert sha(ROOT/cfg['corpus'])==CORPUS_SHA;docs=read(ROOT/cfg['corpus'])['documents'];tokens=read(ROOT/cfg['tokens']);assert len(docs)==2048 and set(tokens)=={d['id'] for d in docs};out=ROOT/cfg['output'];assert out.resolve().drive.upper()=='C:' and out.resolve().is_relative_to(ROOT.resolve());assert shutil.disk_usage(out.parent.resolve()).free>=cfg['minimum_output_disk_free_bytes']
 ops=read(ROOT/cfg['schedule']);assert ops==dict(operations=[dict(id='capture_'+d['id'],kind='forward') for d in docs]);contract=read((ROOT/config).parent/'contract.json');identity=dict(model=sha(ROOT/cfg['model_identity']),code=digest(dict(config=sha(ROOT/config),inputs=sha((ROOT/config).parent/'input_manifest.json'),sources={n:sha(ROOT/n) for n in contract['tracked_files'] if n.endswith('.py')})))
 done=out/'stage_completed.json'
 if done.exists():assert receipt(done)['identity']==identity;return
 runtime=Runtime(out,identity,ops);restore_cached_prefix(runtime,docs,tokens,cfg);pending=[d for d in docs if not(out/'completed'/('capture_'+d['id']+'.json')).exists()];native=None;model=None
 if pending:
  import torch
  from transformers import AutoModelForCausalLM
  from ab_confirmatory_20261101.native import Native
  ident=read(ROOT/cfg['model_identity'])['A']
  for name,h in ident['files'].items():assert sha(ROOT/name)==(h['sha256'] if isinstance(h,dict) else h)
  torch.set_num_threads(1);model,info=AutoModelForCausalLM.from_pretrained(ident['folder'],local_files_only=True,dtype=torch.float32,attn_implementation='eager',low_cpu_mem_usage=True,output_loading_info=True);tied=model.get_input_embeddings().weight.data_ptr()==model.get_output_embeddings().weight.data_ptr();assert not(set(info.get('missing_keys',[]))-({'lm_head.weight'} if tied else set())) and not info.get('mismatched_keys') and not info.get('error_msgs');native=Native(model.eval().float(),17,cfg['answer_ids'])
 try:
  for d in docs:
   def compute(d=d):
    assert native is not None
    v=native.forward(tokens[d['id']]['ids']);return plain({k:v[k][0] for k in ['R','x','y','margin']})
   runtime.compute('capture_'+d['id'],dict(document=d,tokens=tokens[d['id']]),dict(mode='natural'),dict(answer_ids=cfg['answer_ids'],orientation='is-minus-are'),compute)
  immutable(done,dict(identity=identity,forward=2048,backward=0,LLM_loads=1,captures=2048,scope='Only new native captures; no FIT or arm outcomes',qualification_reused=True,output_resolved=str(out.resolve())))
 finally:
  if native is not None:native.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--config');a=p.parse_args();prepare() if a.mode=='prepare' else run(a.config)
