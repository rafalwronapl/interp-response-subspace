"""Fixed-fit inside/outside oracle head readouts; no fitting or model loading."""
import argparse, shutil
from pathlib import Path
import numpy as np
from analysis.C2_fresh_fit_holdout_20261102.worker import read,sha,receipt,plain
from analysis.A_crossed_diagnostic_20261102.diagnostic import current_capture
from ab_confirmatory_20261101.baselines_b import predict_ridge
from analysis.C3_noD_more_models_20261102.evaluate import arrays
ROOT=Path(__file__).resolve().parents[2]
FITS=['r1','r2','r3']
def arms():return ['native']+[k+'_'+i for i in FITS for k in ['P','inside','outside']]
def updates(rec,donor,ps,us):
 D=np.asarray(donor['y'],float)-np.asarray(rec['y'],float);dx=np.asarray(donor['x'],float)-np.asarray(rec['x'],float)
 values={'native':D};energies=[]
 for i in FITS:
  p=predict_ridge(ps[i],dx);e=D-p;u=us[i];c=(e@u)@u.T
  values.update({'P_'+i:p,'inside_'+i:p+c,'outside_'+i:D-c})
  energies.append(dict(fit=i,P=float(e@e),inside=float((e-c)@(e-c)),outside=float(c@c)))
 return values,energies

def run(config):
 import torch
 from research_checks.gate import digest
 from research_checks.persistence import immutable
 from ab_confirmatory_20261101.runtime import Runtime
 cfg=read(ROOT/config);stage=(ROOT/config).parent;out=ROOT/cfg['output'];cap=read(ROOT/cfg['capture_config']);docs=read(ROOT/cfg['corpus'])['documents']
 assert len(docs)==2048 and cfg['arms']==arms();assert out.resolve().drive.upper()=='C:' and shutil.disk_usage(out.parent).free>=1073741824
 contract=read(stage/'contract.json');identity=dict(model=sha(ROOT/cap['model_identity']),code=digest(dict(config=sha(ROOT/config),inputs=sha(stage/'input_manifest.json'),sources={n:sha(ROOT/n) for n in contract['tracked_files'] if n.endswith('.py')})))
 if (out/'stage_completed.json').exists():
  done=receipt(out/'stage_completed.json');assert done['identity']==identity
  assert all(sha(out/n)==h for n,h in done['essential_files'].items());return
 ps={};us={}
 for i in FITS:
  p=next(p for p in cfg['fixed_pairs'] if p['fit']==i)
  for n in ['P','U']:assert sha(ROOT/p[n])==p[n+'_sha256']
  pr=receipt(ROOT/p['P']);ur=receipt(ROOT/p['U']);assert pr['recipe']==dict(kind='MAP',group=i,n=128,regularization=.01);assert ur['recipe']==dict(kind='native_response_U',group=i,n=128,rank=8)
  ps[i]=arrays(pr['fit']);us[i]=np.asarray(ur['U'],float);assert us[i].shape==(640,8)
 torch.set_num_threads(1);a=cap['assets']
 with np.load(ROOT/cap['asset_path'],allow_pickle=False) as z:w={k:torch.tensor(z[k],dtype=torch.float32) for k in ['norm','head']}
 rt=Runtime(out,identity,read(ROOT/cfg['schedule']));rows=[];energies=[]
 for d in docs:
  rec=current_capture(ROOT/cap['output'],d['id'])[0];donor=current_capture(ROOT/cap['output'],d['donor'])[0];vs,es=updates(rec,donor,ps,us)
  meta={k:d[k] for k in ['block','frame','side','q']};context_sha=digest(plain(dict(R=rec['R'],y=rec['y'],updates=vs)))
  # Immutable vector digest sealed before this context's first readout; vectors regenerated deterministically on resume.
  immutable(out/'contexts'/(d['id']+'.json'),dict(document=d['id'],context_sha256=context_sha,fixed_pairs=cfg['fixed_pairs']))
  for arm in cfg['arms']:
   target=(np.asarray(rec['y'],float)+vs[arm]).astype(np.float32)
   def head(target=target,R=rec['R']):
    z=torch.tensor(R,dtype=torch.float32)+torch.tensor(target,dtype=torch.float32);z=z*torch.rsqrt(z.pow(2).mean(-1,keepdim=True)+a['eps']);z=z*(1+w['norm']);logits=torch.nn.functional.linear(z,w['head']);logits=torch.tanh(logits/a['softcap'])*a['softcap'] if a['softcap'] else logits;return float(logits[0]-logits[1])
   value=rt.compute(d['id']+'__'+arm,dict(document=d,context_sha256=context_sha),dict(arm=arm),dict(answer_ids=a['answer_ids'],orientation='is-minus-are'),head)
   assert np.isfinite(value)
   rows.append(dict(document=d['id'],**meta,arm=arm,margin=value))
  energies.extend(dict(document=d['id'],**meta,**e) for e in es)
 immutable(out/'result.json',plain(dict(identity=identity,rows=rows,energies=energies,qualification='Exact accepted C3 head/baseline parity reused; native here is patched donor-response intervention, not natural baseline')))
 # Frozen scalar analysis runs only after all finite head results are immutable.
 # Resume reuses an existing analysis receipt, never repeats valid resampling.
 sources={'result':sha(out/'result.json'),'corpus':sha(ROOT/cfg['corpus']),'analyzer':sha(Path(__file__).parent/'analyze.py')}
 if (out/'analysis_result.json').exists():assert receipt(out/'analysis_result.json')['source_sha256']==sources
 else:
  from analysis.A_gemma270_extension_20261102.analyze import analyze
  summary=analyze(receipt(out/'result.json'),read(ROOT/cfg['corpus']));summary['source_sha256']=sources;immutable(out/'analysis_result.json',plain(summary))
 immutable(out/'stage_completed.json',dict(identity=identity,forward=20480,backward=0,LLM_loads=0,scalar_analysis='Frozen1999draw block analysis;0additionalmodelcalls',essential_files={n:sha(out/n) for n in ['result.json','analysis_result.json']}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)
