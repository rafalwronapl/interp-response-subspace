"""FIT-only regressions; SELECT-only choices; sealed TEST prediction interface."""
import numpy as np
from .common import payload,load_arrays

LAMBDAS=(1e-4,.01,1.)
RANKS=(1,2,4,8)

def inputs(x20,x21,scale):
    return np.concatenate([x20/scale[0],x21/scale[1]],axis=1)/np.sqrt(2.)

def ridge_maps(x,y):
    n=len(x);g=x@x.T;v,q=np.linalg.eigh(g)
    return {lam:x.T@(q@((q.T@y)/(v[:,None]+n*lam))) for lam in LAMBDAS}

def low_fit(x,y,p,xs,ds,ps):
    _,singular,vt=np.linalg.svd(y,full_matrices=False)
    maps=ridge_maps(x,y);best=None
    for rank in RANKS:
        u=vt[:rank].T
        for lam in sorted(LAMBDAS,reverse=True):
            w=maps[lam]@u;error=float(np.mean(np.sum((ds-(ps+xs@w@u.T))**2,axis=1)))
            if best is None or error < best[0]-1e-12*max(abs(error),abs(best[0])):
                best=(error,rank,lam,w.copy(),u.copy())
    return best,singular,vt

def fit(fit,select):
    d,p=fit['D'].astype(float),fit['P'].astype(float)
    ds,ps=select['D'].astype(float),select['P'].astype(float)
    scale=np.asarray([np.sqrt(np.mean(np.sum(fit[k].astype(float)**2,axis=1))) for k in ('dx20','dx21')])
    if not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('Invalid FIT input scale')
    x=inputs(fit['dx20'],fit['dx21'],scale);xs=inputs(select['dx20'],select['dx21'],scale)
    residual=d-p
    best,singular,vt=low_fit(x,residual,p,xs,ds,ps)
    direct,_,_=low_fit(x,d,np.zeros_like(p),xs,ds,np.zeros_like(ps))
    maps=ridge_maps(x,residual);full=None
    for lam in sorted(LAMBDAS,reverse=True):
        err=float(np.mean(np.sum((ds-ps-xs@maps[lam])**2,axis=1)))
        if full is None or err<full[0]-1e-12*max(abs(err),abs(full[0])):full=(err,lam,maps[lam])
    denominator=float(np.sum(p*p))
    if denominator<=0:raise ValueError('Undefined FIT scalar baseline')
    arrays=dict(scale=scale,C_W=best[3],C_U=best[4],B4_W=full[2],B5_W=direct[3],B5_U=direct[4],
                alpha=np.asarray(np.sum(p*d)/denominator),B3_mean=np.stack([residual[fit['cell']==i].mean(axis=0) for i in range(8)]),
                singular=singular,fit_U=vt[:8].T)
    # Each group of 32 consecutive rows is one whole lexical block.
    if len(x)!=384:raise ValueError('FIT row count')
    rng=np.random.Generator(np.random.PCG64(36037));permutations=[]
    for i in range(20):
        order=rng.permutation(12);permutations.append(order.tolist())
        r=residual.reshape(12,32,-1)[order].reshape(384,-1)
        # Recompute SVD and coefficients for each permutation, as frozen in plan.
        _,_,v=np.linalg.svd(r,full_matrices=False);u=v[:best[1]].T
        w=ridge_maps(x,r@u)[best[2]]
        arrays[f'perm{i}_W']=w;arrays[f'perm{i}_U']=u
    meta=dict(candidate=dict(rank=best[1],lambda_=best[2],select_error=best[0]),
              direct=dict(rank=direct[1],lambda_=direct[2],select_error=direct[0]),
              full=dict(lambda_=full[1],select_error=full[0]),permutations=permutations,permutation_seed=36037,
              refit_on_select=False)
    return arrays,meta

def predict(parameters,data):
    # This allowlist deliberately excludes D/R, logits, local target and readout effects.
    required={'dx20','dx21','P','cell','B6'}
    if set(data)!=required:raise ValueError('Prediction input allowlist violation')
    x=inputs(data['dx20'],data['dx21'],parameters['scale']);p=data['P'].astype(float)
    result=dict(C=p+x@parameters['C_W']@parameters['C_U'].T,
        B0=np.zeros_like(p),B1=p,B2=float(parameters['alpha'])*p,
        B3=p+parameters['B3_mean'][data['cell']],B4=p+x@parameters['B4_W'],
        B5=x@parameters['B5_W']@parameters['B5_U'].T,B6=data['B6'].astype(float))
    for i in range(20):result[f'perm{i}']=p+x@parameters[f'perm{i}_W']@parameters[f'perm{i}_U'].T
    if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('Nonfinite forecast')
    return result

def signal(data,epsilon,required):
    results=[]
    for block in sorted(set(data['block'].tolist())):
        for branch in (0,1):
            for dose in (.5,1.):
                ix=(data['block']==block)&(data['branch']==branch)&(data['dose']==dose)
                d=data['D'][ix].astype(float);r=d-data['P'][ix]
                ed=float(np.mean(np.sum(d*d,axis=1)));er=float(np.mean(np.sum(r*r,axis=1)))
                ratio=None if ed<=0 else er/ed
                ok=ratio is not None and ratio>=.1 and np.sqrt(ed)>100*epsilon and np.sqrt(er)>100*epsilon
                results.append(dict(block=int(block),branch=branch,dose=dose,energy_D=ed,energy_R=er,ratio=ratio,ok=bool(ok)))
    counts={f'{b}_{dose}':sum(r['ok'] for r in results if r['branch']==b and r['dose']==dose) for b in (0,1) for dose in (.5,1.)}
    return dict(ready=all(n>=required for n in counts.values()),required_blocks=required,counts=counts,cells=results,epsilon_num=epsilon)

def assess(data,predictions,epsilon,rank,fit_U):
    d=data['D'].astype(float);blocks=sorted(set(data['block'].tolist()))
    if len(blocks)!=12 or len(d)!=384:raise ValueError('TEST independent units')
    loss={name:np.asarray([np.mean(np.sum((d[data['block']==b]-v[data['block']==b])**2,axis=1)) for b in blocks]) for name,v in predictions.items()}
    rng=np.random.Generator(np.random.PCG64(36036));ix=rng.integers(0,12,size=(20000,12))
    boot={k:v[ix].mean(axis=1) for k,v in loss.items()};means={k:float(v.mean()) for k,v in loss.items()}
    signal_result=signal(data,epsilon,9);comparisons={};undefined=False
    for name,tau in [('B0',.1),('B1',.2),('B2',.1),('B3',.1)]:
        denominator=boot[name];point=means[name]
        valid=point>0 and np.all(denominator>0) and np.isfinite(denominator).all()
        if not valid:undefined=True
        gains=1-boot['C']/denominator if valid else None
        diff=boot[name]-boot['C']
        comparisons[name]=dict(gain=1-means['C']/point if valid else None,threshold=tau,
            gain_lower=float(np.quantile(gains,.05,method='linear')) if valid else None,
            gain_upper=float(np.quantile(gains,.95,method='linear')) if valid else None,
            raw_diff_lower=float(np.quantile(diff,.05,method='linear')),raw_diff_upper=float(np.quantile(diff,.95,method='linear')))
    residual=d-data['P'];undefined=undefined or np.mean(np.sum(d*d,axis=1))<=0 or np.mean(np.sum(residual*residual,axis=1))<=0
    marginal={}
    for kind,values in [('branch',(0,1)),('dose',(.5,1.))]:
        for value in values:
            mask=data[kind]==value
            marginal[f'{kind}_{value}']=float(np.mean(np.sum((d[mask]-predictions['B1'][mask])**2,axis=1)-np.sum((d[mask]-predictions['C'][mask])**2,axis=1)))
    if undefined or not signal_result['ready']:decision='uninformative'
    elif all(v['gain']>=v['threshold'] and v['raw_diff_lower']>0 for v in comparisons.values()) and all(v>0 for v in marginal.values()):decision='supported'
    elif any(v['gain_upper']<v['threshold'] for v in comparisons.values()):decision='not_supported'
    else:decision='undetermined'
    energy=float(np.mean(np.sum(d*d,axis=1)));metrics={}
    for name,v in predictions.items():
        nv=float(np.linalg.norm(v));nd=float(np.linalg.norm(d))
        metrics[name]=dict(MSE=means[name],NRMSE=float(np.sqrt(means[name]/energy)) if energy>0 else None,
            global_cosine=float(np.sum(d*v)/(nv*nd)) if nv>0 and nd>0 else None,predicted_to_actual_rms=nv/nd if nd>0 else None)
    r4=None if np.any(boot['B4']<=0) else float(np.quantile(boot['C']/boot['B4'],.95,method='linear'))
    delta5=boot['B5']-boot['C'];gain5=None if means['B5']<=0 else 1-means['C']/means['B5']
    u=fit_U[:,:rank];projection=residual@u@u.T
    return dict(decision=decision,metrics=metrics,comparisons=comparisons,marginal_B1_improvement=marginal,signal=signal_result,
        low_rank_vs_full=dict(ratio_upper95=r4,supported=r4 is not None and r4<=1.10),
        additional_CLT_value=dict(gain=gain5,raw_difference_lower95=float(np.quantile(delta5,.05,method='linear')),
            supported=gain5 is not None and gain5>=.1 and float(np.quantile(delta5,.05,method='linear'))>0),
        posthoc_projection=dict(relative_error=float(np.linalg.norm(residual-projection)/np.linalg.norm(residual)) if np.linalg.norm(residual)>0 else None,
             uses_true_TEST_residual=True,is_forecast=False),
        block_loss={k:v.tolist() for k,v in loss.items()},bootstrap=dict(draws=20000,seed=36036,unit='whole lexical block',descriptive=True))
