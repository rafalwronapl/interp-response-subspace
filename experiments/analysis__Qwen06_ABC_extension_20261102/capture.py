from analysis.C2_fresh_fit_holdout_20261102.worker import *
def run(config):
 from research_checks.gate import digest
 from research_checks.persistence import immutable
 from ab_confirmatory_20261101.runtime import Runtime
 from ab_confirmatory_20261101.baselines_b import compact_ridge
 from ab_confirmatory_20261101.fitting import basis
 import torch
 from transformers import AutoModelForCausalLM
 cfg=read(ROOT/config);modelname=cfg['model'];docs=read(ROOT/cfg['corpus'])['documents'];tokens=read(ROOT/cfg['tokens']);out=ROOT/cfg['output'];contract=read((ROOT/config).parent/'contract.json');sources={n:sha(ROOT/n) for n in contract['tracked_files']};identity=dict(model=sha(ROOT/cfg['model_identity']),code=digest(dict(config=sha(ROOT/config),sources=sources)))
 assert str(out.resolve())==cfg['physical_output'];assert shutil.disk_usage(out.resolve()).free>cfg['resource_min_free_disk_bytes'];ident=read(ROOT/cfg['model_identity']);ident=ident if modelname=='qwen' else ident['A']
 for name,h in ident['files'].items():assert sha(ROOT/name)==(h['sha256'] if isinstance(h,dict) else h),'Checkpoint changed'
 runtime=Runtime(out,identity,read(ROOT/cfg['schedule']));caps={};torch.set_num_threads(1)
 model,loading=AutoModelForCausalLM.from_pretrained(ident['folder'],local_files_only=True,dtype=torch.float32,attn_implementation='eager',low_cpu_mem_usage=True,output_loading_info=True)
 tied=model.get_input_embeddings().weight.data_ptr()==model.get_output_embeddings().weight.data_ptr();assert not(set(loading.get('missing_keys',[]))-({'lm_head.weight'} if tied else set())) and not loading.get('mismatched_keys') and not loading.get('error_msgs')
 if modelname=='qwen':from a_short_qwen_20261101.native import Native
 else:from ab_confirmatory_20261101.native import Native
 native=Native(model.eval().float(),-1,cfg['assets']['answer_ids'])
 def call(op,d,setting,fn):return runtime.compute(op,dict(document=d,tokens=tokens[d['id']]),setting,dict(answer_ids=cfg['assets']['answer_ids'],orientation='is-minus-are'),fn)
 def capture(d):
  def fn():
   v=native.forward(tokens[d['id']]);return plain({k:v[k][0] for k in ['R','x','y','margin']})
  return call('capture_'+d['id'],d,dict(mode='natural'),fn)
 initial=[d for d in docs if d['id'] in cfg['controls']]
 for d in initial:caps[d['id']]=capture(d)
 controls=[]
 for d in initial:
  r=caps[d['id']];donor=caps[d['donor']];ids=tokens[d['id']]
  def full(label,mode,value,position=-1):return call('control_'+d['id']+'_'+label,d,dict(mode=mode,value=plain(value),position=position),lambda:plain(native.forward(ids,mode,value,full_head=True,position=position)))
  sx=full('sham_input','input',r['x']);sy=full('sham_output','output',r['y']);h=full('full_head',None,None);dn=full('donor_input','input',donor['x']);w=model.get_output_embeddings().weight[cfg['assets']['answer_ids']].detach();direction=w[0]-w[1];direction/=direction.norm();pos=full('positive','output',np.asarray(r['y'])+.1*direction.numpy());neg=full('negative','output',np.zeros_like(r['y']),-2);suffix=call('control_'+d['id']+'_suffix',d,dict(mode='suffix'),lambda:float(native.suffix(r['R'],r['y'])[0]))
  parity=max(abs(sx['margin'][0]-r['margin']),abs(sy['margin'][0]-r['margin']),abs(h['margin'][0]-r['margin']),abs(neg['margin'][0]-r['margin']),abs(suffix-r['margin']));branch=float(np.linalg.norm(np.asarray(dn['y'][0])-donor['y'])/np.linalg.norm(donor['y']));sensitivity=abs(pos['margin'][0]-r['margin']);passed=parity<=1e-4 and branch<=1e-4 and sensitivity>=1e-6 and np.linalg.norm(neg['patch_before'])>0;controls.append(dict(document=d['id'],parity=parity,branch=branch,sensitivity=sensitivity,passed=bool(passed)))
 immutable(out/'controls.json',controls);assert all(c['passed'] for c in controls),'Technical qualification failed'
 for d in docs:
  if d['id'] not in caps:caps[d['id']]=capture(d)
 native.close();del native,model,w;gc.collect()
 immutable(out/"stage_completed.json",dict(identity=identity,forward=2844,backward=0,LLM_loads=1,captures=len(caps),scope="Capture-only; no FIT or heldout intervention outcomes"))

if __name__=="__main__":
 import sys
 run(sys.argv[1])
