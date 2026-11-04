"""Bounded last-layer CLT input-dose study; native suffix and all cells retained."""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
import sys,json,shutil,numpy as np,torch
from pathlib import Path
from analysis.C2_fresh_fit_holdout_20261102.worker import ROOT,read,sha,receipt,plain,analytic_gradient
from analysis.A_crossed_diagnostic_20261102.diagnostic import current_capture
from analysis.Qwen06_ABC_extension_20261102.b_math import load_assets,branch,branch_jvp,clt_response,suffix
from ab_confirmatory_20261101.baselines_b import compact_ridge,predict_ridge
from analysis.Qwen06_ABC_extension_20261102.bundle_runtime import BundleRuntime as Runtime
from research_checks.persistence import immutable
from research_checks.gate import digest
DOSES=[.05,.1,.25,.5,1.,2.];METHODS=['CLT','TAYLOR','COMPOSED_LINEAR','FROZEN','RIDGE','MEAN','ZERO']

def run(config):
 cfg=read(ROOT/config);stage=(ROOT/config).parent;out=ROOT/cfg['output'];assert out.resolve().is_relative_to(ROOT) and shutil.disk_usage(out.parent).free>1073741824;torch.set_num_threads(1)
 capcfg=read(ROOT/cfg['capture_config']);capture=ROOT/capcfg['output'];docs=read(ROOT/capcfg['corpus'])['documents'];fitdocs=[d for d in docs if d['role']=='FIT' and d['cohort']=='r1' and d['split']=='FIT_MAP'];holdout=[d for d in docs if d['role']=='HOLDOUT'];assert len(fitdocs)==128 and len(holdout)==2048
 a=load_assets();headmeta=capcfg['assets']
 with np.load(ROOT/capcfg['asset_path']) as f:norm=torch.tensor(f['norm'],dtype=torch.float32);head=torch.tensor(f['head'],dtype=torch.float32)
 contract=read(stage/'contract.json');identity=dict(model=sha(ROOT/capcfg['model_identity']),code=digest(dict(config=sha(ROOT/config),input_manifest=sha(stage/'input_manifest.json'),sources={n:sha(ROOT/n) for n in contract['tracked_files']})))
 runtime=Runtime(out,identity,read(ROOT/cfg['schedule']));capture_memo={}
 def get(doc):
  if doc not in capture_memo:capture_memo[doc]=current_capture(capture,doc)[0]
  return capture_memo[doc]
 # Charge extracted-weight parity and analytic composed-derivative controls.
 controls=[];adouble=dict(a,native={k:v.double() for k,v in a['native'].items()})
 for doc in [d for d in fitdocs if d['id'] in capcfg['controls']]:
  raw=get(doc['id']);donor=get(doc['donor']);x=torch.tensor(raw['x'],dtype=torch.float32);dx=torch.tensor(np.asarray(donor['x'])-raw['x'],dtype=torch.float64);y=torch.tensor(raw['y'],dtype=torch.float32);R=torch.tensor(raw['R'],dtype=torch.float32);operations=['control_'+doc['id']+'_'+s for s in ['branch','head','branch_plus','branch_minus','branch_center64','head_plus','head_minus']]
  def control(x=x,dx=dx,y=y,R=R,raw=raw):
   actual=branch(x,a);margin=suffix(R,y,torch.zeros_like(y),norm,head,headmeta['eps']);eps=1e-4;plus=branch(x.double()+eps*dx,adouble);minus=branch(x.double()-eps*dx,adouble);numeric=(plus-minus)/(2*eps);analytic=branch_jvp(x.double(),dx,adouble)
   center64=branch(x.double(),adouble);relative=float(torch.linalg.vector_norm(actual-y)/torch.linalg.vector_norm(y));parity=abs(float(margin)-raw['margin']);derivative=float(torch.linalg.vector_norm(analytic-numeric)/max(float(torch.linalg.vector_norm(analytic)),1e-12));J=analytic_gradient((R.double()+center64).numpy(),norm.double().numpy(),head.double().numpy(),headmeta['eps'],None,False)
   # Composed derivative numerical check in FP64 preserves the actual norm/head
   # function and captures both branch derivative and downstream readout geometry.
   def double_readout(branch_y):
    z=R.double()+branch_y;z=z/torch.sqrt(torch.mean(z*z)+headmeta['eps']);logits=torch.nn.functional.linear(z*norm.double(),head.double());return float(logits[0]-logits[1])
   composed=(double_readout(plus)-double_readout(minus))/(2*eps);pred=float(J@analytic.numpy());comp_error=abs(composed-pred)/max(abs(composed),abs(pred),1e-8)
   return dict(branch_relative=relative,head_absolute=parity,JVP_relative=derivative,composed_relative=comp_error,native_JVP_norm=float(torch.linalg.vector_norm(analytic)),composed_derivative=pred,passed=bool(relative<=1e-4 and parity<=1e-4 and derivative<=1e-5 and comp_error<=1e-4 and np.isfinite(pred)))
  result=runtime.compute(operations[0],dict(document=doc),dict(mode='technical_branch_head_JVP'),dict(answer_ids=[374,525]),control,extra_operations=operations[1:]);controls.append(result)
 immutable(out/'controls.json',controls);assert len(controls)==4 and all(c['passed'] for c in controls) and max(c['native_JVP_norm'] for c in controls)>1e-6 and max(abs(c['composed_derivative']) for c in controls)>1e-6,'B extracted-branch/JVP qualification or nonzero reachability failed'
 del adouble
 # Train each non-unit-dose ridge on its OWN fixed FIT-only native labels.
 X=[];Xc={str(c):[] for c in DOSES};Ys={str(c):[] for c in DOSES};classes={str(c):{} for c in DOSES}
 for doc in fitdocs:
  raw=get(doc['id']);donor=get(doc['donor']);x=np.asarray(raw['x']);dx=np.asarray(donor['x'])-x;X.append(dx);other=[c for c in DOSES if c!=1.];ops=['fit_'+doc['id']+'_c'+str(c) for c in other]
  def fit_branches(raw=raw,x=x,dx=dx):return branch(torch.tensor(np.array([x+c*dx for c in other]),dtype=torch.float32),a).double().numpy().tolist()
  values=runtime.compute(ops[0],dict(document=doc),dict(mode='FIT_native_dose_labels',doses=other),dict(scope='Vector labels only; no heldout labels'),fit_branches,extra_operations=ops[1:]);label={c:np.asarray(v)-raw['y'] for c,v in zip(other,values)};label[1.]=np.asarray(donor['y'])-raw['y']
  for c in DOSES:Xc[str(c)].append((x+c*dx).astype(np.float32).astype(float)-x);Ys[str(c)].append(label[c]);classes[str(c)].setdefault(str(doc['side'])+'_'+str(doc['q']),[]).append(label[c])
 X=np.asarray(X);fits={};means={}
 for c in DOSES:
  key=str(c);target=np.asarray(Ys[key]);path=out/'fits'/('c'+key+'.json');recipe=dict(dose=c,cohort='r1',split='FIT_MAP',n=128,regularization=.01,condition_mean='side_q')
  if path.exists():v=receipt(path);assert v['identity']==identity and v['recipe']==recipe;fits[key]=v['fit'];means[key]=v['means']
  else:
   fit=receipt(ROOT/cfg['unit_dose_fit'])['fit'] if c==1. else compact_ridge(np.asarray(Xc[key]),target,.01)
   mean={k:np.mean(v,axis=0) for k,v in classes[key].items()};immutable(path,plain(dict(identity=identity,recipe=recipe,fit=fit,means=mean)));fits[key]=fit;means[key]=mean
 immutable(out/'FIT_frozen.json',dict(identity=identity,files={str(p.relative_to(out)):sha(p) for p in (out/'fits').glob('*.json')},heldout_B_outcomes_used=False))
 fit_sha=sha(out/'FIT_frozen.json');old=receipt(ROOT/cfg['aligned_evaluation_result']);adopted={}
 for row in old['rows']:
  if row['arm'] in ['native','P_r1']:adopted[row['document'],row['arm']]=row['margin']
 rows=[]
 for doc in holdout:
  raw=get(doc['id']);donor=get(doc['donor']);x=np.asarray(raw['x']);dx=np.asarray(donor['x'])-x;y=np.asarray(raw['y']);R=np.asarray(raw['R']);ops=holdout_operations(doc['id'])
  def compute(raw=raw,donor=donor,doc=doc,x=x,dx=dx,y=y,R=R):
   xt=torch.tensor(x,dtype=torch.float32);xc=torch.tensor(np.array([x+c*dx for c in DOSES]),dtype=torch.float32);other=[i for i,c in enumerate(DOSES) if c!=1.];native=branch(xc[other],a).double().numpy()-y;native_by={i:v for i,v in zip(other,native)};native_by[4]=np.asarray(donor['y'])-y
   tc,frozen_dir=clt_response(xt,xc,torch.tensor(dx,dtype=torch.float32),a);jvp=branch_jvp(xt,torch.tensor(dx,dtype=torch.float32),a).double().numpy();J=analytic_gradient((torch.tensor(R,dtype=torch.float32)+torch.tensor(y,dtype=torch.float32)).double().numpy(),norm.double().numpy(),head.double().numpy(),headmeta['eps'],None,False);local=float(J@jvp);result=[]
   for k,c in enumerate(DOSES):
    delta=native_by[k];key=str(c);preds={'CLT':tc[k].double().numpy(),'TAYLOR':(np.float32(c)*jvp.astype(np.float32)).astype(float),'FROZEN':(np.float32(c)*frozen_dir.numpy()).astype(float),'RIDGE':predict_ridge(fits[key],xc[k].double().numpy()-x),'MEAN':np.asarray(means[key][str(doc['side'])+'_'+str(doc['q'])])};margins={method:float(suffix(torch.tensor(R,dtype=torch.float32),torch.tensor(y,dtype=torch.float32),torch.tensor(p,dtype=torch.float64),norm,head,headmeta['eps'])) for method,p in preds.items() if not(method=='RIDGE' and c==1.)};margins['RIDGE']=adopted[doc['id'],'P_r1'] if c==1. else margins['RIDGE'];true=adopted[doc['id'],'native'] if c==1. else float(suffix(torch.tensor(R,dtype=torch.float32),torch.tensor(y,dtype=torch.float32),torch.tensor(delta,dtype=torch.float64),norm,head,headmeta['eps']));H=true-raw['margin'];predicted={m:margins[m]-raw['margin'] for m in preds};predicted.update(COMPOSED_LINEAR=c*local,ZERO=0.)
    result.append(dict(dose=c,native_effect=H,predictions=predicted,native_vector_sq=float(delta@delta),CLT_vector_error_sq=float(np.sum((tc[k].double().numpy()-delta)**2))))
   return result
  result=runtime.compute(ops[0],dict(document=doc),dict(mode='heldout_full_input_doses',doses=DOSES,FIT_frozen_sha256=fit_sha),dict(answer_ids=[374,525],orientation='is-minus-are'),compute,extra_operations=ops[1:]);rows.extend(dict(document=doc['id'],block=doc['block'],frame=doc['frame'],side=doc['side'],q=doc['q'],**r) for r in result)
 immutable(out/'result.json',plain(dict(identity=identity,rows=rows,FIT_frozen_sha256=fit_sha,scope='PublicCLTsource27component/localinputdoses, conceptual extension, one fulloperator; no qualifiedpopulationclaims')));immutable(out/'stage_completed.json',dict(identity=identity,forward=96924,backward=0,LLM_loads=0,adopted_unit_native_and_ridge_heads=4096,analytic_JVP=2052,holdout_blocks=256,holdout_contexts=2048))

def holdout_operations(doc):
 return [doc+'__CLT_BASE_ENCODER',doc+'__FROZEN_DIRECTION']+[doc+'__'+kind+'_c'+str(c) for c in DOSES for kind in ['CLT','head_CLT','head_TAYLOR','head_FROZEN','head_MEAN']+(['branch','head_native','head_RIDGE'] if c!=1. else [])]
if __name__=='__main__':run(sys.argv[1])
