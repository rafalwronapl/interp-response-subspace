"""The eight frozen E036 formulas; target-free allowlist, single FP32 cast."""
import numpy as np
from .common import METHODS

def predict(parameters,data):
    if set(data)!={'dx20','dx21','P','cell','B6'}:raise ValueError('Prediction allowlist')
    scale=parameters['scale']
    x=np.concatenate([data['dx20']/scale[0],data['dx21']/scale[1]],axis=1)/np.sqrt(2.)
    p=data['P'].astype(float)
    result=dict(C=p+x@parameters['C_W']@parameters['C_U'].T,B0=np.zeros_like(p),B1=p,
        B2=float(parameters['alpha'])*p,B3=p+parameters['B3_mean'][data['cell']],
        B4=p+x@parameters['B4_W'],B5=x@parameters['B5_W']@parameters['B5_U'].T,B6=data['B6'])
    if set(result)!=set(METHODS) or not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError('Nonfinite/missing prediction')
    return {k:np.asarray(result[k],dtype=np.float32) for k in METHODS}
