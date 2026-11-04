"""Fixed linear/RBF kernel ridge and block-separated selection."""
import numpy as np
PARAMS=[dict(kernel=k,lam=l) for k in ['linear','rbf'] for l in [1.,.01,.0001,.000001]]
def distances(x,z):return np.maximum(np.sum(x*x,1)[:,None]+np.sum(z*z,1)[None,:]-2*x@z.T,0.)
def fit(x,y,param):
    x=np.asarray(x,float);y=np.asarray(y,float);xm=x.mean(0);ym=y.mean(0);a=x-xm
    scale=None;degenerate=False
    if param['kernel']=='linear':k=a@a.T
    else:
        ds=distances(a,a);positive=ds[np.triu_indices(len(ds),1)];positive=positive[positive>0]
        scale=float(np.median(positive)) if len(positive) else 1.;degenerate=not len(positive)
        k=np.exp(-ds/scale)
    alpha=np.linalg.solve(k+len(x)*param['lam']*np.eye(len(x)),y-ym)
    return dict(xmean=xm,ymean=ym,train=a,alpha=alpha,scale=scale,degenerate=degenerate,param=param)
def predict(m,x):
    a=np.asarray(x,float)-m['xmean']
    k=a@m['train'].T if m['param']['kernel']=='linear' else np.exp(-distances(a,m['train'])/m['scale'])
    return k@m['alpha']+m['ymean']
def folds(blocks,n=4):
    ids=sorted(set(blocks));return [np.isin(blocks,ids[j::n]) for j in range(n)]
def loss(y,p,blocks):return float(np.mean([np.mean(np.sum((y[blocks==b]-p[blocks==b])**2,1)) for b in sorted(set(blocks))]))
def select(x,y,blocks,linear_only=False):
    params=PARAMS[:4] if linear_only else PARAMS;scores=[]
    for param in params:
        predicted=np.empty_like(y,dtype=float)
        for valid in folds(blocks):
            m=fit(x[~valid],y[~valid],param);predicted[valid]=predict(m,x[valid])
        scores.append(dict(**param,block_L2=loss(y,predicted,blocks)))
    # Candidate order supplies predeclared exact-tie policy.
    chosen=int(np.argmin([s['block_L2'] for s in scores]))
    return params[chosen],scores
