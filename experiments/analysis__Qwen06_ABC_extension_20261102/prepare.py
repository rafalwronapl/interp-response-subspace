"""Prepare Qwen0.6 native-capture packet; zero pretrained forwards."""
import inspect,json,hashlib,runpy
from pathlib import Path
from analysis.C2_fresh_fit_holdout_20261102 import worker as c2
from analysis.A_crossed_diagnostic_20261102 import diagnostic as old
ROOT=c2.ROOT;BASE=Path(__file__).parent;put=c2.put;sha=c2.sha;read=c2.read
FOLDER=ROOT/'model'

def main():
 from transformers import AutoTokenizer
 stage=BASE/'capture_base_v2';stage.mkdir(exist_ok=True);rel=stage.relative_to(ROOT).as_posix()
 identity=stage/'model_identity.json'
 if not identity.exists():
  files={str(p):sha(p) for p in sorted(FOLDER.iterdir()) if p.is_file()}
  original=read(ROOT/'analysis/E041_execution/identity.json')['model_identity']['native'];assert files[str(FOLDER/'model.safetensors')]==original['weights_sha256']
  put(identity,dict(folder=str(FOLDER),repo=original['id'],revision=original['revision'],files=files,layer=27,hidden=1024,load_dtype='float32',stored_dtype='bfloat16',architecture='qwen3',not_E041_L21=True,source='analysis/E041_execution/identity.json'))
 master=ROOT/'analysis/AB_confirmatory_execution_20261101/corpus/corpus_v1.json'
 shared=ROOT/'analysis/A_gemma270_extension_20261102/corpus.json'
 assert shared.exists(),'Waiting only for shared immutable corpus publication'
 fit=[dict(d,role='FIT') for d in read(master)['documents'] if d['study']=='A' and d['cohort'] in ['r1','r2','r3'] and d['split'] in ['FIT_MAP','FIT_U']]
 hold=[dict(d,role='HOLDOUT') for d in read(shared)['documents']]
 assert len(fit)==768 and len(hold)==2048
 docs=fit+hold;ids={d['id'] for d in docs};assert len(ids)==2816 and all(d['donor'] in ids for d in docs)
 tok=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True);answers=[tok.encode(s,add_special_tokens=False) for s in [' is',' are']];assert answers==[[374],[525]],answers
 tokens={d['id']:tok(d['text'])['input_ids'] for d in docs};assert all(1<len(t)<=256 and tok.unk_token_id not in t for t in tokens.values())
 for n,v in [('corpus.json',dict(documents=docs)),('tokens.json',tokens)]:put(stage/n,v)
 source=inspect.getsource(old.extract_assets);ns=dict(vars(old),OUT=stage);exec(source,ns)
 assets=ns['extract_assets']('qwen',dict(model_identity=rel+'/model_identity.json',tokens=rel+'/tokens.json'))
 controls=[d['id'] for d in fit if d['cohort']=='r1' and d['split']=='FIT_MAP' and d['index']==0 and d['frame']=='F0'];assert len(controls)==4
 operations=[dict(id='capture_'+d['id'],kind='forward') for d in docs]+[dict(id='control_'+i+'_'+k,kind='forward') for i in controls for k in ['sham_input','sham_output','full_head','donor_input','positive','negative','suffix']]
 assert len(operations)==2844
 cfg=dict(model='qwen',corpus=rel+'/corpus.json',tokens=rel+'/tokens.json',model_identity=rel+'/model_identity.json',output=rel+'/run',schedule=rel+'/operations.json',controls=controls,assets=assets,asset_path=rel+'/qwen_head_assets.npz',resource_min_free_disk_bytes=1073741824,threads=1)
 put(stage/'operations.json',dict(operations=operations));put(stage/'config.json',cfg)
 # Literal frozen capture function from qualified C2, stopping before fitting.
 source=inspect.getsource(c2.run).split('\n ps,us=anchors(modelname)')[0]
 source+='\n immutable(out/"stage_completed.json",dict(identity=identity,forward=2844,backward=0,LLM_loads=1,captures=len(caps),scope="Capture-only; no FIT or heldout intervention outcomes"))\n'
 (BASE/'capture.py').write_text('from analysis.C2_fresh_fit_holdout_20261102.worker import *\n'+source+'\nif __name__=="__main__":\n import sys\n run(sys.argv[1])\n',encoding='utf-8')
 helper='analysis/AB_confirmatory_execution_20261101/close_short_pilot_imports_v1.py';seeds=[BASE.relative_to(ROOT).as_posix()+'/'+n for n in ['capture.py','prepare.py','PROTOCOL.txt']]+[helper,'a_short_qwen_20261101/native.py'];tracked=sorted(set(seeds)|set(runpy.run_path(str(ROOT/helper))['closure'](seeds)))
 sources=[master,shared,identity,stage/'corpus.json',stage/'tokens.json',stage/'qwen_head_assets.npz',stage/'config.json',stage/'operations.json',ROOT/'analysis/AB_confirmatory_execution_20261101/corpus/observed_words_v2.json']
 put(stage/'input_manifest.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in sources}))
 put(stage/'evidence.json',dict(prepared=True,full_capture=2816,full_controls=24,suffix_controls=4,forward=2844,backward=0,LLM_loads=1,answer_ids=[374,525],scope='Model-specific fresh Qwen06 responses; shared lexical blocks; no adaptation from TEST results'))
 put(stage/'contract.json',dict(schema_version=1,id='Qwen06_ABC_capture_v1',kind='experiment',primary_agent='/root/sol_qwen06_parallel',purpose='Prospective native capture for fixed A/C protocols, three new fixed-fit groups and256fresh blocks, last L27 not historicalE041L21.',tracked_files=tracked,tracked_trees=[],input_manifests=[rel+'/input_manifest.json'],gates=[dict(id=k,category=k,evidence=rel+'/evidence.json',pointer='/prepared',op='eq',expected=True,reason='Qualified native adapter, charged four controls before scale; exact tokenizer/identity/data/exposure/restart evidence inspected independently.') for k in ['hooks','data','pilot','accounting','restart']],circuits=[],structural_scope='Full native capture finalL27 rawMLP; is-minus-are; actual technical controls precede scale.',entrypoint=dict(module='analysis.Qwen06_ABC_extension_20261102.capture',args=[rel+'/config.json']),schedule=rel+'/operations.json',budget=dict(forward=2844,backward=0)))
 print('Capture packet ready:2844F,1load,0B')
if __name__=='__main__':main()
