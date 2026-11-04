"""C3 vector-only residual learning; no-D seal precedes true holdout-D access."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
from analysis.C2_fresh_fit_holdout_20261102.worker import read,sha,receipt,plain,analytic_gradient
from analysis.A_crossed_diagnostic_20261102.diagnostic import current_capture
from ab_confirmatory_20261101.baselines_b import compact_ridge,predict_ridge
from ab_confirmatory_20261101.fitting import basis
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent;FITS=['r1','r2','r3']

def arrays(f):
 return {k:np.asarray(v,float) if k in ['a','dual','zm','ym'] else v for k,v in f.items()}

def extract_noD_features(recipient_raw,donor_raw):
 return dict(R=recipient_raw['R'],x=recipient_raw['x'],y=recipient_raw['y'],donor_x=donor_raw['x'])

def noD_predict(features,ps,residual,pooled,us,assets):
 """Pure boundary: NO document links, donor y, D, margin or true outcomes."""
 assert set(features)=={'R','x','y','donor_x'},'Forbidden/no-D feature field'
 f={k:np.asarray(v,float) for k,v in features.items()};d=len(f['x']);assert all(v.shape==(d,) and np.isfinite(v).all() for v in f.values())
 a,norm,head,gemma=assets;dx=f['donor_x']-f['x'];updates={'baseline':np.zeros(d)};gradients={};rows=[]
 for i in FITS:
  p=predict_ridge(ps[i],dx);updates['P_'+i]=p;z=(f['R'].astype(np.float32)+(f['y']+p).astype(np.float32)).astype(float);J=analytic_gradient(z,norm,head,a['eps'],a['softcap'],gemma);gradients['J_'+i]=J
  for j in FITS:
   rhat=predict_ridge(residual[i,j],dx);u=us[j];c=(rhat@u)@u.T;jr=float(J@rhat);jc=float(J@c);score=float(jc*jc-2*jr*jc);fullscore=float(-jr*jr)
   updates['direct_'+i+'_'+j]=predict_ridge(pooled[i,j],dx);updates['full_'+i+'_'+j]=p+rhat;updates['proj_'+i+'_'+j]=p+c
   rows.append(dict(i=i,j=j,score_projected=score,score_full=fullscore,apply_projected=bool(score<0),apply_full=bool(fullscore<0),estimated_residual_norm=float(np.linalg.norm(rhat)),estimated_projected_norm=float(np.linalg.norm(c)),J_rhat=jr,J_c=jc))
 return updates,gradients,rows

def publish_npz(path,values):
 import tempfile
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 def check():
  with np.load(path,allow_pickle=False) as z:assert set(z.files)==set(values) and all(np.array_equal(z[k],v) for k,v in values.items()),'Immutable vector data changed'
 if path.exists():check();return
 fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
 try:
  with os.fdopen(fd,'wb') as stream:np.savez(stream,**values);stream.flush();os.fsync(stream.fileno())
  try:os.link(tmp,path)
  except FileExistsError:check()
 finally:Path(tmp).unlink(missing_ok=True)


def vector_hash(values):
 h=hashlib.sha256()
 for k in sorted(values):
  a=np.ascontiguousarray(values[k]);h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()

def run(config):
 import torch
 from research_checks.gate import digest
 from research_checks.persistence import immutable
 from analysis.Qwen06_ABC_extension_20261102.bundle_runtime import BundleRuntime as Runtime
 cfg=read(ROOT/config);stage=(ROOT/config).parent;out=ROOT/cfg['output'];capture=ROOT/cfg['capture_stage'];capcfg=read(capture/'config.json');docs=read(ROOT/cfg['corpus'])['documents'];assert cfg['model']=='qwen06_base';assert cfg['arms']==expected_arms()
 assert str(out.resolve())==cfg['physical_output'];assert shutil.disk_usage(out.resolve()).free>1073741824;torch.set_num_threads(1)
 contract=read(stage/'contract.json');identity=dict(model=digest(cfg['assets']),code=digest(dict(config=sha(ROOT/config),inputs=sha(stage/'input_manifest.json'),sources={n:sha(ROOT/n) for n in contract['tracked_files'] if n.endswith('.py')})))
 if (out/'stage_completed.json').exists():
  done=receipt(out/'stage_completed.json');assert done['identity']==identity
  for name,h in done['essential_files'].items():assert sha(out/name)==h
  return
 runtime=Runtime(out,identity,read(ROOT/cfg['schedule']))
 capture_memo={}
 def get(doc):
  if doc not in capture_memo:capture_memo[doc]=current_capture(ROOT/capcfg['output'],doc)[0]
  return capture_memo[doc]
 # FIT-only label dictionary. No holdout response arrays enter any training.
 training={d['id']:get(d['id']) for d in docs if d['role']=='FIT'};holdout=[d for d in docs if d['role']=='HOLDOUT'];assert len(holdout)==2048
 def matrices(cohort,split):
  ds=[d for d in docs if d['cohort']==cohort and d['split']==split and d['role']=='FIT'];assert len(ds)==128
  return np.array([np.asarray(training[d['donor']]['x'])-training[d['id']]['x'] for d in ds]),np.array([np.asarray(training[d['donor']]['y'])-training[d['id']]['y'] for d in ds])
 groups={i:{s:matrices(i,s) for s in ['FIT_MAP','FIT_U']} for i in FITS};ps={};us={};residual={};pooled={};fit_hashes={}
 def fit_file(name,x,y,recipe):
  p=out/'fits'/(name+'.json')
  if p.exists():v=receipt(p);assert v['identity']==identity and v['recipe']==recipe;fit=v['fit']
  else:fit=compact_ridge(x,y,.01);immutable(p,plain(dict(identity=identity,recipe=recipe,fit=fit)))
  fit_hashes[p.relative_to(out).as_posix()]=sha(p);return arrays(fit)
 for i in FITS:
  x,y=groups[i]['FIT_MAP'];ps[i]=fit_file('P_'+i,x,y,dict(kind='MAP',group=i,n=128,regularization=.01))
 for j in FITS:
  x,y=groups[j]['FIT_U'];p=out/'fits'/('U_'+j+'.json');recipe=dict(kind='native_response_U',group=j,n=128,rank=8)
  if p.exists():v=receipt(p);assert v['identity']==identity and v['recipe']==recipe;us[j]=np.asarray(v['U'],float)
  else:us[j]=basis(y,8);immutable(p,plain(dict(identity=identity,recipe=recipe,U=us[j])))
  fit_hashes[p.relative_to(out).as_posix()]=sha(p)
 for i in FITS:
  for j in FITS:
   ux,uy=groups[j]['FIT_U'];mx,my=groups[i]['FIT_MAP'];label=uy-predict_ridge(ps[i],ux)
   residual[i,j]=fit_file('residual_'+i+'_'+j,ux,label,dict(kind='residual_on_disjoint_U',P=i,U=j,n=128,regularization=.01))
   pooled[i,j]=fit_file('pooled_'+i+'_'+j,np.concatenate([mx,ux]),np.concatenate([my,uy]),dict(kind='pooled_direct',MAP=i,U=j,n=256,regularization=.01))
 a=cfg['assets']
 with np.load(ROOT/cfg['asset_path'],allow_pickle=False) as f:norm=f['norm'].astype(float);head=f['head'].astype(float);weights={k:torch.tensor(f[k],dtype=torch.float32) for k in ['norm','head']}
 assets=(a,norm,head,cfg['model']=='gemma270');noD_records=[];noD_contexts=[];features_dir=out/'noD_vectors';features_dir.mkdir(parents=True,exist_ok=True)
 for d in holdout:
  # Read only donor x into pure feature pipeline. No donor y/D/margin access.
  r=get(d['id']);features=extract_noD_features(r,get(d['donor']));donor_x=features['donor_x']
  updates,gradients,records=noD_predict(features,ps,residual,pooled,us,assets)
  for row in records:row.update({k:d[k] for k in ['block','frame','side','q']});row['document']=d['id'];noD_records.append(row)
  file=features_dir/(d['id']+'.npz');values=dict(R=np.asarray(r['R'],np.float32),y=np.asarray(r['y'],np.float32),x_rec=np.asarray(r['x']),x_donor=np.asarray(donor_x),**updates,**gradients)
  noD_contexts.append(dict(document=d['id'],vector_sha256=vector_hash(values)))
 # FIRST immutable publication: no-D vectors/forecasts, before holdout true-D.
 immutable(out/'noD_forecast.json',plain(dict(identity=identity,records=noD_records,contexts=noD_contexts,fit_hashes=fit_hashes,feature_boundary=['recipient_R','recipient_x','recipient_y','donor_x'],holdout_D_constructed_or_used=False,raw_record_deserialization=True,finite_intervention_head_calls_so_far=0,unique_gradient_points=6144,analytic_gradient_evaluations_so_far=6144)))
 noD_sha=sha(out/'noD_forecast.json');assert len(receipt(out/'noD_forecast.json')['records'])==18432

 def reproduce_noD(d,context):
  r=get(d['id']);updates,gradients,_=noD_predict(extract_noD_features(r,get(d['donor'])),ps,residual,pooled,us,assets)
  v=dict(R=np.asarray(r['R'],np.float32),y=np.asarray(r['y'],np.float32),x_rec=np.asarray(r['x']),x_donor=np.asarray(get(d['donor'])['x']),**updates,**gradients)
  assert vector_hash(v)==context['vector_sha256'];return v
 def reproduce_oracle(d,n,context):
  D=np.asarray(get(d['donor'])['y'])-get(d['id'])['y'];v={'native':D}
  for i in FITS:
   p=n['P_'+i];res=D-p
   for j in FITS:
    c=(res@us[j])@us[j].T;v['oracle_'+i+'_'+j]=p+c;v['outside_'+i+'_'+j]=D-c
  assert vector_hash(v)==context['vector_sha256'];return v

 oracle_records=[];oracle_contexts=[];oracle_dir=out/'oracle_vectors';oracle_dir.mkdir(exist_ok=True)
 for d,context in zip(holdout,noD_contexts):
  # ONLY AFTER no-D seal, true donor y and D become available for reference.
  assert (out/'noD_forecast.json').exists();r=get(d['id']);donor_y=get(d['donor'])['y'];D=np.asarray(donor_y)-r['y']
  original=reproduce_noD(d,context)
  values={'native':D}
  for i in FITS:
   p=original['P_'+i];res=D-p;J=original['J_'+i]
   for j in FITS:
    c=(res@us[j])@us[j].T;values['oracle_'+i+'_'+j]=p+c;values['outside_'+i+'_'+j]=D-c;score=float((J@c)**2-2*(J@res)*(J@c));oracle_records.append(dict(document=d['id'],block=d['block'],frame=d['frame'],side=d['side'],q=d['q'],i=i,j=j,score=score,energy_P=float(res@res),energy_inside=float(c@c),energy_outside=float((res-c)@(res-c))))
  file=oracle_dir/(d['id']+'.npz')
  oracle_contexts.append(dict(document=d['id'],vector_sha256=vector_hash(values)))
 # SECOND publication: oracle reference, never passed back into no-D policy.
 immutable(out/'oracle_forecast.json',dict(identity=identity,noD_forecast_sha256=noD_sha,records=oracle_records,contexts=oracle_contexts,finite_intervention_head_calls_so_far=0,scope='Oracle reference/evaluation only; no D data enters no-D policy'))
 oracle_sha=sha(out/'oracle_forecast.json');rows=[]
 for d,ncontext,ocontext in zip(holdout,noD_contexts,oracle_contexts):
  n=reproduce_noD(d,ncontext);o=reproduce_oracle(d,n,ocontext)
  def evaluate_context(n=n,o=o):
   values={}
   for arm in cfg['arms']:
    update=n[arm] if arm in n else o[arm];target=(n['y'].astype(float)+update).astype(np.float32)
    z=torch.tensor(n['R'],dtype=torch.float32)+torch.tensor(target,dtype=torch.float32);z=z*torch.rsqrt(z.pow(2).mean(-1,keepdim=True)+a['eps']);z=z*((1+weights['norm']) if cfg['model']=='gemma270' else weights['norm']);logits=torch.nn.functional.linear(z,weights['head']);logits=torch.tanh(logits/a['softcap'])*a['softcap'] if a['softcap'] else logits;values[arm]=float(logits[0]-logits[1])
   assert all(np.isfinite(v) for v in values.values()),'Nonfinite finite-head output'
   assert abs(values['baseline']-get(d['id'])['margin'])<=1e-4,'Captured native baseline/head parity failed'
   return values
  operations=[d['id']+'__'+arm for arm in cfg['arms']]
  values=runtime.compute(operations[0],dict(document=d,context_sha256=ncontext['vector_sha256'],oracle_context_sha256=ocontext['vector_sha256']),dict(arms=cfg['arms'],noD_forecast_sha256=noD_sha,oracle_forecast_sha256=oracle_sha),dict(answer_ids=a['answer_ids'],orientation='is-minus-are'),evaluate_context,extra_operations=operations[1:])
  for arm in cfg['arms']:rows.append(dict(document=d['id'],block=d['block'],frame=d['frame'],side=d['side'],q=d['q'],arm=arm,margin=values[arm]))
 # THIRD publication: finite outcomes, only now eligible for utility scoring.
 immutable(out/'result.json',dict(identity=identity,rows=rows,noD_forecast_sha256=noD_sha,oracle_forecast_sha256=oracle_sha,scope='No-D policy was sealed before true holdout-D/oracle references and finite outcomes'))
 essential={p.name:sha(p) for p in [out/'noD_forecast.json',out/'oracle_forecast.json',out/'result.json']};immutable(out/'stage_completed.json',dict(identity=identity,forward=102400,backward=0,LLM_loads=0,unique_gradient_points=6144,analytic_gradient_evaluations=18432,essential_files=essential))

def expected_arms():
 return ['baseline','native']+['P_'+i for i in FITS]+[kind+'_'+i+'_'+j for kind in ['direct','full','proj','oracle','outside'] for i in FITS for j in FITS]
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(a.config)
