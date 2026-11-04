"""Paired whole-block inference, separate construction decisions, no fitting."""
import numpy as np
from .common import METHODS


def quantile(a,p):return float(np.quantile(a,p,method='linear'))


def comparison(a,b,ba,bb):
    valid=bool(b>0 and np.isfinite(b) and np.all(bb>0) and np.isfinite(bb).all())
    gain=1-ba/bb if valid else None
    delta=bb-ba
    return dict(defined=valid,gain=float(1-a/b) if valid else None,
        gain_lower=quantile(gain,.05) if valid else None,gain_upper=quantile(gain,.95) if valid else None,
        raw_difference_lower=quantile(delta,.05),raw_difference_upper=quantile(delta,.95))


def decide(comps,thresholds,cells,technical,signal_ready,absolute=None):
    if not technical:return 'technically_unresolved'
    if not signal_ready or any(not comps[k]['defined'] for k in thresholds):return 'uninformative'
    absolute_ok=absolute is None or (absolute['point']<=0 and absolute['upper']<=0)
    if all(comps[k]['gain']>=thresholds[k] and comps[k]['raw_difference_lower']>0 for k in thresholds) and all(v>0 for v in cells) and absolute_ok:
        return 'supported'
    if any(comps[k]['gain_upper']<thresholds[k] for k in thresholds) or (absolute is not None and absolute['lower']>0):
        return 'not_supported'
    return 'undetermined'


def joint(states):
    for state in ('technically_unresolved','not_supported','uninformative'):
        if state in states:return state
    return 'supported' if all(s=='supported' for s in states) else 'undetermined'


def signal(data,epsilon,required):
    h=np.asarray(data['Ftrue'],dtype=float)-np.asarray(data['B0'],dtype=float)
    rows=[];frames={}
    for frame in (0,1,2):
        for block in sorted(set(data['block'].tolist())):
            for branch in (0,1):
                for dose in (.5,1.):
                    mask=(data['block']==block)&(data['frame']==frame)&(data['branch']==branch)&(data['dose']==dose)
                    if np.sum(mask)!=4:raise ValueError('Signal cell size')
                    rms=float(np.sqrt(np.mean(h[mask]**2)))
                    rows.append(dict(block=int(block),frame=frame,branch=branch,dose=dose,RMS_H=rms,
                        ok=bool(rms>100*epsilon and rms>=.01)))
        counts={f'{br}_{d}':sum(r['ok'] for r in rows if r['frame']==frame and r['branch']==br and r['dose']==d)
            for br in (0,1) for d in (.5,1.)}
        frames[str(frame)]=dict(ready=all(n>=required for n in counts.values()),counts=counts)
    return dict(ready=all(f['ready'] for f in frames.values()),frames=frames,required=required,epsilon_logit=epsilon,blocks=rows)


def vector_diagnostics(data,vectors):
    d=np.asarray(data['D'],dtype=float)
    def summarize(mask):
        actual=d[mask];norm=float(np.linalg.norm(actual));result={}
        for method in METHODS:
            v=np.asarray(vectors[method][mask],dtype=float);nv=float(np.linalg.norm(v))
            result[method]=dict(MSE=float(np.mean(np.sum((actual-v)**2,axis=1))),
                NRMSE=float(np.linalg.norm(actual-v)/norm) if norm>0 else None,
                cosine=float(np.sum(actual*v)/(norm*nv)) if norm>0 and nv>0 else None,
                norm_ratio=nv/norm if norm>0 else None)
        return result
    pooled=summarize(np.ones(len(d),dtype=bool));groups={}
    for key in ('block','frame','S','A','branch','dose'):
        groups[key]={str(value):dict(rows=int(np.sum(data[key]==value)),metrics=summarize(data[key]==value))
            for value in sorted(set(data[key].tolist()))}
    return pooled,groups


def assess(data,epsilon,technical=True,draws=20000):
    n=len(data['Ftrue']);blocks=sorted(set(data['block'].tolist()))
    if len(blocks)!=12 or n!=576:raise ValueError('MAIN dimensions')
    ftrue=data['Ftrue'].astype(float);fnat=data['Fnat'].astype(float)
    if not all(np.isfinite(data[k]).all() for k in ('Ftrue','Fnat',*METHODS)):raise ValueError('Nonfinite readout')
    # Verify the Cartesian design, rather than allowing duplicates to count as replication.
    seen=set(zip(*(data[k].tolist() for k in ('block','frame','S','A','branch','dose'))))
    expected={(b,f,s,a,br,d) for b in blocks for f in (0,1,2) for s in (0,1) for a in (0,1) for br in (0,1) for d in (.5,1.)}
    if seen!=expected:raise ValueError('Cartesian condition coverage')
    sq={k:(ftrue-data[k].astype(float))**2 for k in METHODS}
    ix=np.random.Generator(np.random.PCG64(39039)).integers(0,12,size=(draws,12))
    sig=signal(data,epsilon,9);frames={};means_by_frame={};boots_by_frame={}
    h=ftrue-data['B0'];t=ftrue-fnat
    thresholds=dict(B0=.1,B1=.2,B2=.1,B3=.1)
    for frame in (0,1,2):
        mask=data['frame']==frame
        losses={k:np.array([np.mean(v[mask&(data['block']==b)]) for b in blocks]) for k,v in sq.items()}
        means={k:float(v.mean()) for k,v in losses.items()};boot={k:v[ix].mean(axis=1) for k,v in losses.items()}
        means_by_frame[frame]=means;boots_by_frame[frame]=boot
        metrics={};energy_h=means['B0'];energy_t=float(np.mean(t[mask]**2))
        for k in METHODS:
            measurable=mask&(np.abs(t)>100*epsilon)
            pred=data[k]-fnat
            metrics[k]=dict(MSE=means[k],RMSE=float(np.sqrt(means[k])),MSE_lower=quantile(boot[k],.05),MSE_upper=quantile(boot[k],.95),
                NRMSE_H=float(np.sqrt(means[k]/energy_h)) if energy_h>0 else None,
                NRMSE_T=float(np.sqrt(means[k]/energy_t)) if energy_t>0 else None,
                sign_eligible=int(np.sum(measurable)),sign_abstained=int(np.sum(mask&~measurable)),
                sign_accuracy=float(np.mean(np.sign(pred[measurable])==np.sign(t[measurable]))) if np.any(measurable) else None,
                preference_agreement=float(np.mean((data[k][mask]>0)==(ftrue[mask]>0))))
        transfers={}
        for candidate in ('B5','C'):
            comps={k:comparison(means[candidate],means[k],boot[candidate],boot[k]) for k in thresholds}
            cells={}
            for br in (0,1):
                for dose in (.5,1.):
                    m=mask&(data['branch']==br)&(data['dose']==dose)
                    cells[f'{br}_{dose}']=float(np.mean(sq['B1'][m]-sq[candidate][m]))
            delta=boot[candidate]-.25*boot['B0']
            absolute=dict(point=means[candidate]-.25*energy_h,lower=quantile(delta,.05),upper=quantile(delta,.95))
            transfers[candidate]=dict(decision=decide(comps,thresholds,cells.values(),technical,sig['frames'][str(frame)]['ready'],absolute),
                comparisons=comps,cell_improvement=cells,absolute_error_constraint=absolute)
        pair={}
        for candidate,other in (('B5','C'),('C','B5')):
            comp=comparison(means[candidate],means[other],boot[candidate],boot[other]);cells={}
            for br in (0,1):
                for dose in (.5,1.):
                    m=mask&(data['branch']==br)&(data['dose']==dose)
                    cells[f'{br}_{dose}']=float(np.mean(sq[other][m]-sq[candidate][m]))
            pair[candidate+'_leads']=dict(decision=decide({other:comp},{other:.1},cells.values(),technical,sig['frames'][str(frame)]['ready']),
                comparison=comp,cell_improvement=cells)
        frames[str(frame)]=dict(role='reference' if frame==0 else 'transfer',metrics=metrics,transfers=transfers,**pair,
            energy_H=energy_h,energy_T=energy_t,RMS_H=float(np.sqrt(energy_h)),RMS_T=float(np.sqrt(energy_t)),
            natural_accuracy=float(np.mean((fnat[mask]>0)==(data['S'][mask]==1))),block_loss={k:v.tolist() for k,v in losses.items()})
    degradation={}
    for frame in (1,2):
        result={}
        for k in METHODS:
            a,b=means_by_frame[frame],means_by_frame[0];ba,bb=boots_by_frame[frame],boots_by_frame[0]
            valid=bool(a['B0']>0 and b['B0']>0 and np.all(ba['B0']>0) and np.all(bb['B0']>0))
            if valid:
                g=ba[k]/ba['B0']-bb[k]/bb['B0']
                result[k]=dict(defined=True,point=a[k]/a['B0']-b[k]/b['B0'],lower=quantile(g,.05),upper=quantile(g,.95),degradation_supported=quantile(g,.05)>0)
            else:result[k]=dict(defined=False,point=None,lower=None,upper=None,degradation_supported=None)
        degradation[str(frame)]=result
    breakdown={}
    for key in ('block','frame','S','A','branch','dose'):
        breakdown[key]={str(v):dict(rows=int(np.sum(data[key]==v)),MSE={k:float(np.mean(sq[k][data[key]==v])) for k in METHODS},
            RMS_H=float(np.sqrt(np.mean(h[data[key]==v]**2)))) for v in sorted(set(data[key].tolist()))}
    return dict(H1_joint=joint([frames[str(f)]['transfers']['B5']['decision'] for f in (1,2)]),
        frames=frames,paired_degradation=degradation,signal=sig,technical_controls_ok=bool(technical),breakdown=breakdown,
        bootstrap=dict(draws=draws,seed=39039,unit='whole lexical block shared across frames',quantiles=[.05,.95],descriptive=True))
