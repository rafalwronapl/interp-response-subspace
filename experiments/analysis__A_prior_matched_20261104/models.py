"""Whole-block twofold CV; readout-independent vector loss only."""
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.A_L21_learned_20261103.models import PARAMS,fit,predict,loss
from analysis.A_L21_learned_20261103.common import packed,array
def pack(m):return {k:packed(v) if isinstance(v,np.ndarray) else v for k,v in m.items()}
def unpack(m):return {k:array(v) if isinstance(v,dict) and set(v)=={'dtype','shape','data'} else v for k,v in m.items()}
@threadpool_limits.wrap(limits=2)
def train(x,D,P,U,blocks):
 models={};scores={}
 for name,y in [('C',(D-P)@U),('same_U',D@U),('Direct',D),('Direct_P',D-P)]:
  ss=[]
  for param in PARAMS:
   predicted=np.empty_like(y,dtype=float)
   for b in sorted(set(blocks)):
    valid=blocks==b;predicted[valid]=predict(fit(x[~valid],y[~valid],param),x[valid])
   ss.append(dict(param=param,loss=loss(y,predicted,blocks)))
  chosen=int(np.argmin([s['loss'] for s in ss]));models[name]=pack(fit(x,y,PARAMS[chosen]));scores[name]=dict(candidates=ss,selected=chosen)
 return dict(models=models,scores=scores)
def forecast(x,prior,bundles):
 U=prior['B5_U'];P=x@prior['B5_W']@U.T;out=dict(P=np.asarray(P,np.float32).astype(float))
 for c,bundle in enumerate(bundles):
  for method,m in bundle['models'].items():
   raw=predict(unpack(m),x);out[f'{method}_{c}']=out['P']+raw@U.T if method=='C' else raw@U.T if method=='same_U' else out['P']+raw if method=='Direct_P' else raw
 return out
