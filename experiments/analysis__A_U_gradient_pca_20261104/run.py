"""Posthoc A controls on the fresh L21 TEST (64 blocks, 1024 conditions).

1) share of vector error e=D-P inside span(B5_U);
2) native readout gradient g at the predicted state (L22-27 suffix, autograd):
   share of gradient energy in span(U) vs random expectation r/d;
3) oracle repair with other label-free rank-8 spaces (PCA of natural L21
   outputs, centered/uncentered; class-mean difference + PCA), scored
   through the exact native suffix, versus B5_U.
Uses cached native records only for states; loads the model once for the
suffix. No refitting of P/U.
"""
import sys,json,glob,os,time
from pathlib import Path
ROOT=Path(r'C:\Users\rafal\interpretability-lab');sys.path.insert(0,str(ROOT))
import numpy as np,torch
from ab_confirmatory_20261101.native_tail_b import decode
from e031.loading import load
OUT=Path(__file__).resolve().parent;RUN=ROOT/'analysis/AB_five_experiments_20261103/A_L21/main_run'
log=open(OUT/'log.txt','a',encoding='utf-8')
def say(*a):
    s=' '.join(map(str,a));print(s,flush=True);log.write(s+'\n');log.flush()
def st(rec):return {k:(decode(v).numpy() if isinstance(v,dict) else np.asarray(v)) for k,v in rec['state'].items()}

say('start',time.ctime())
nat={};tgt={};fitD=[]
for f in glob.glob(str(RUN/'cache/*.json')):
    if os.path.getsize(f)<5000:continue
    d=json.load(open(f))['payload'];r=d['request'];role=r['intervention'].get('role');doc=r['document']
    if role=='natural':nat[doc['id']]=(doc,d['result'])
    elif role=='TEST_native_target':
        iv=r['intervention'];tgt[(doc['id'],iv['branch'],float(iv['dose']))]=(doc,d['result'])
say('natural',len(nat),'test targets',len(tgt))
old=dict(np.load(ROOT/'runs/E036/development/predictors.npz'));U=old['B5_U'].astype(float)
fc=json.load(open(RUN/'TEST_forecasts.json'))['payload']
def cid(doc,b,dz):return f"{doc['id']}_{b}_{dz}"
# natural L21 outputs
Ynat={k:st(v[1])['y'].astype(float).ravel() for k,v in nat.items()}
fit_ids=[k for k,(doc,_) in nat.items() if doc['split'].startswith('FIT')]
Y=np.stack([Ynat[k] for k in fit_ids]);say('FIT natural',Y.shape)
def topk(M,k):return np.linalg.svd(M,full_matrices=False)[2][:k].T
S_=np.array([nat[k][0]['S'] for k in fit_ids])
spaces={'B5_U':U,'PCA_nat_uncentered':topk(Y,8),'PCA_nat_centered':topk(Y-Y.mean(0),8)}
md=(Y[S_==1].mean(0)-Y[S_==0].mean(0));md/=np.linalg.norm(md)
R=Y-Y.mean(0);R=R-np.outer(R@md,md);Q,_=np.linalg.qr(np.column_stack([md,topk(R,7)]));spaces['classdiff+PCA7']=Q
rng=np.random.default_rng(4110431);spaces['random8']=np.linalg.qr(rng.standard_normal((U.shape[0],8)))[0]
torch.set_num_threads(4);model=load(ROOT/'model');model.eval()
W=model.lm_head.weight[[374,525]].detach()
def suffix(rec,ys,grad=False):
    h=decode(rec['H']);r=decode(rec['R']);v=torch.as_tensor(np.asarray(ys,np.float32),dtype=h.dtype)
    if grad:v.requires_grad_(True)
    hh=h.expand(len(v),-1,-1).clone();hh[:,-1]=r[0,-1]+v
    ctx=torch.enable_grad() if grad else torch.inference_mode()
    with ctx:
        for L in range(22,28):hh=model.model.layers[L](hh,**decode(rec['kwargs'][str(L)]))
        z=torch.nn.functional.linear(model.model.norm(hh)[:,-1:],W);m=z[:,0,1]-z[:,0,0]
        if grad:
            m.sum().backward();return m.detach().double().numpy(),v.grad.double().numpy()
    return m.double().numpy(),None
rows=[];t0=time.time()
for i,((did,b,dz),(doc,rec)) in enumerate(sorted(tgt.items())):
    s=st(nat[did][1]);t=st(rec);D=(t['y']-s['y']).astype(float).ravel()
    P=decode(fc[cid(doc,b,dz)]['old']).numpy().astype(float).ravel() if cid(doc,b,dz) in fc else None
    if P is None:raise KeyError(cid(doc,b,dz))
    y0=s['y'].astype(float).ravel();e=D-P
    truth=suffix(rec,[t['y'].ravel()])[0][0];mp,g=suffix(rec,[y0+P],grad=True);g=g[0]
    cand={k:y0+P+Q@(Q.T@e) for k,Q in spaces.items()}
    names=list(cand);ms=suffix(rec,[cand[k] for k in names])[0]
    row=dict(id=did,block=doc['block'],branch=b,dose=dz,truth=float(truth),P=float(mp[0]),
             e_in_U=float(np.sum((U.T@e)**2)/np.sum(e**2)),g_in_U=float(np.sum((U.T@g)**2)/np.sum(g**2)),
             lin_e=float(g@e),lin_e_in_U=float(g@(U@(U.T@e))),
             **{'g_in_'+k:float(np.sum((Q.T@g)**2)/np.sum(g**2)) for k,Q in spaces.items()},
             **{'m_'+k:float(v) for k,v in zip(names,ms)})
    rows.append(row)
    if i%64==0:say(i,len(tgt),f'{time.time()-t0:.0f}s')
json.dump(rows,open(OUT/'rows.json','w'))
import collections
blk=collections.defaultdict(list)
for r in rows:blk[r['block']].append(r)
def bmse(key):return float(np.mean([np.mean([(r[key]-r['truth'])**2 for r in v]) for v in blk.values()]))
base=bmse('P');res=dict(conditions=len(rows),blocks=len(blk),MSE_P=base,
 repair={k:dict(MSE=bmse('m_'+k),reduction_pct=100*(1-bmse('m_'+k)/base)) for k in spaces},
 mean_e_share_in_U=float(np.mean([r['e_in_U'] for r in rows])),
 grad_share={k:float(np.mean([r['g_in_'+k] for r in rows])) for k in spaces},random_expectation=8/U.shape[0],
 linear_readout_error_share_in_U=float(np.sum([r['lin_e_in_U']*r['lin_e'] for r in rows])/np.sum([r['lin_e']**2 for r in rows])))
json.dump(res,open(OUT/'summary.json','w'),indent=1);say(json.dumps(res,indent=1));say('done',time.ctime())
