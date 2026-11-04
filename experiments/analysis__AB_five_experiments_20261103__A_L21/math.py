"""FIT-only B5 recipe and separate exact versus norm-rescaled oracle controls."""
import numpy as np
RANKS=(1,2,4,8,16,32)
def predict(p,dx20,dx21):
 x=np.concatenate([dx20/p['scale'][0],dx21/p['scale'][1]],axis=-1)/np.sqrt(2.)
 return np.asarray(x@p['B5_W']@p['B5_U'].T,dtype=np.float32)
def fit(dx20,dx21,D,seed):
 scale=np.array([np.sqrt(np.mean(np.sum(x.astype(float)**2,axis=1))) for x in [dx20,dx21]])
 if np.any(scale<=0):raise ValueError('Undefined FIT scale')
 x=np.concatenate([dx20/scale[0],dx21/scale[1]],axis=1)/np.sqrt(2.)
 _,_,vt=np.linalg.svd(D.astype(float),full_matrices=False);U=vt[:32].T
 lam=1e-4;W=x.T@np.linalg.solve(x@x.T+len(x)*lam*np.eye(len(x)),D.astype(float)@U[:,:8])
 random=np.linalg.qr(np.random.default_rng(seed).normal(size=(D.shape[1],32)))[0]
 return dict(scale=scale,B5_W=W,B5_U=U[:,:8],U=U,random_U=random)
def corrections(error,U,V):
 a=U@(U.T@error);b=V@(V.T@error);na=np.linalg.norm(a);nb=np.linalg.norm(b)
 # A zero/near-zero direction is undefined; no arbitrary large multiplier.
 tiny=64*np.finfo(float).eps*max(1.,np.linalg.norm(error))
 c=None if nb<=tiny else b*(na/nb)
 return a,b,c
