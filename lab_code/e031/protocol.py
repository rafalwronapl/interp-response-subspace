"""Scalar formulas and primary decision for E031, without model/gradient execution."""
import numpy as np

BASELINES=('zero','P1','mean_linear','mean_quadratic','source_linear')


def predictions(q, delta, dose, b_prime, c_prime, b_second, c_second, mu, beta):
    values=[q,delta,dose,b_prime,c_prime,b_second,c_second,mu,beta]
    if q not in (-1,1) or dose not in (.5,1.) or not np.isfinite(values).all():
        raise ValueError('Invalid fixed-dose forecast inputs')
    p1=q*dose*(b_prime-c_prime)
    # Derivatives are with respect to t where u(t)=u0+t*delta: no extra delta here.
    return dict(P1=p1,P2=p1+q*dose*dose*(b_second-c_second)/2,zero=0.,
                mean_linear=mu*dose,mean_quadratic=mu*dose*dose,source_linear=beta*q*dose*delta)


def summarize(cases):
    cc=[c for c in cases if c['dose']==1.]
    blocks=sorted({c['group'] for c in cc})
    if len(blocks)!=16 or len(cc)!=128:raise ValueError('Expected 16 complete blocks at dose1')
    wanted={(f,s,a) for f in ('F0','F1') for s in (0,1) for a in (0,1)}
    for g in blocks:
        subset=[c for c in cc if c['group']==g]
        if len(subset)!=8 or {(c['frame'],c['S'],c['A']) for c in subset}!=wanted:
            raise ValueError('Incomplete/duplicate F,S,A')
    if not all(np.isfinite([c['C']]+list(c['predictions'].values())).all() for c in cc):
        raise ValueError('Nonfinite case')
    ix=np.random.default_rng(31031).integers(0,16,size=(16384,16))
    def arr(fn):return np.array([np.mean([fn(c) for c in cc if c['group']==g]) for g in blocks])
    def ci(v):return np.quantile(v[ix].mean(axis=1),[.0125,.9875]).tolist()
    signal=arr(lambda c:abs(c['C']));G=float(signal.mean())
    groups={(f,s):[c for c in cc if (c['frame'],c['S'])==(f,s)] for f in ('F0','F1') for s in (0,1)}
    groupG={k:float(np.mean([abs(c['C']) for c in v])) for k,v in groups.items()}
    informative=G>=.02 and ci(signal)[0]>.01 and min(groupG.values())>=.01 and sum(signal>=.02)>=8
    errors={k:arr(lambda c:abs(c['predictions'][k]-c['C'])) for k in ('P2',)+BASELINES}
    a=ci(errors['P2']);r=ci(errors['P2']-.25*signal)
    guard=all(np.mean([abs(c['predictions']['P2']-c['C']) for c in v])<=min(.01,.25*groupG[k]) for k,v in groups.items())
    supported=a[1]<.01 and r[1]<0 and guard
    excluded=a[0]>.01 or r[0]>0
    comparisons={}
    for k in BASELINES:
        improvement=errors[k]-errors['P2'];absolute=ci(improvement);relative=ci(improvement-.2*errors[k])
        supported &= absolute[0]>.002 and relative[0]>0
        excluded |= absolute[1]<.002 or relative[1]<0
        comparisons[k]=dict(MAE=float(errors[k].mean()),improvement_CI=absolute,relative_excess_CI=relative)
    return dict(status='uninformative' if not informative else 'supported' if supported else 'not_supported' if excluded else 'undetermined',
                informative=bool(informative),G=G,MAE=float(errors['P2'].mean()),MAE_CI=a,
                excess_error_CI=r,subgroup_guard=bool(guard),comparisons=comparisons)
