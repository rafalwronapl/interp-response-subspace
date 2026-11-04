"""Posthoc A control 3 (matched strength); see ctrl2 for the donor-value variant.
Posthoc A control 2 on the fresh L21 TEST (64 blocks, 1024 conditions).

Response space of OTHER interventions not selected by any readout: three
seeded random layer-19 SwiGLU neurons (excluding 1317) moved toward their
donor values on the 96 FIT documents (branch B, doses .5/1); rank-8 SVD of
the stacked native L21 responses, as for the historical U.
Also: no-P repair y0+QQ^T D; gradient ceiling (top-8 PCA of native readout
gradients at FIT native endpoints); 20 random rank-8 spaces.
Every repair is scored through the exact native L22-27 suffix.
"""
import sys,json,glob,os,time,collections
from pathlib import Path
ROOT=Path(r'C:\Users\rafal\interpretability-lab');sys.path.insert(0,str(ROOT))
import numpy as np,torch
from ab_confirmatory_20261101.native_tail_b import decode
from e031.loading import load
from e036.subject import Subject
OUT=Path(__file__).resolve().parent;RUN=ROOT/'analysis/AB_five_experiments_20261103/A_L21/main_run'
log=open(OUT/'ctrl3_log.txt','a',encoding='utf-8')
def say(*a):
    s=' '.join(map(str,a));print(s,flush=True);log.write(s+'\n');log.flush()
def st(rec):return {k:(decode(v).numpy() if isinstance(v,dict) else np.asarray(v)) for k,v in rec['state'].items()}
say('start',time.ctime())
docs={d['id']:d for d in json.load(open(ROOT/'analysis/AB_five_experiments_20261103/A_L21/corpus.json'))['documents']}
fit_docs=[d for d in docs.values() if d['split'].startswith('FIT')]
nat={};tgt={};fitt={}
for f in glob.glob(str(RUN/'cache/*.json')):
    if os.path.getsize(f)<5000:continue
    d=json.load(open(f))['payload'];r=d['request'];role=r['intervention'].get('role');doc=r['document']
    if role=='natural':nat[doc['id']]=d['result']
    elif role=='TEST_native_target':iv=r['intervention'];tgt[(doc['id'],iv['branch'],float(iv['dose']))]=d['result']
    elif role=='FIT_native_target':iv=r['intervention'];fitt[(doc['id'],iv['branch'],float(iv['dose']))]=d['result']
say('natural',len(nat),'test',len(tgt),'fit targets',len(fitt))
old=dict(np.load(ROOT/'runs/E036/development/predictors.npz'));U=old['B5_U'].astype(float);dim=U.shape[0]
fc=json.load(open(RUN/'TEST_forecasts.json'))['payload']
torch.set_num_threads(4);model=load(ROOT/'model');model.eval()
W=model.lm_head.weight[[374,525]].detach()
inter=model.model.layers[19].mlp.down_proj.weight.shape[1]
rng=np.random.default_rng(20261104);neurons=[int(k) for k in rng.choice([i for i in range(inter) if i!=1317],3,replace=False)]
say('random neurons',neurons)
# 1) other-intervention response spaces, MATCHED STRENGTH
# target: mean norm of historical (neuron 1317) native FIT responses, branch B
tn=[]
for (did,b,dz),rec in fitt.items():
    if b=='B':tn.append(float(np.linalg.norm(st(rec)['y'].astype(float).ravel()-st(nat[did])['y'].astype(float).ravel())))
target={.5:float(np.mean([v for (k,v) in zip([k for k in fitt if k[1]=='B'],tn) if k[2]==.5])),1.:float(np.mean([v for (k,v) in zip([k for k in fitt if k[1]=='B'],tn) if k[2]==1.]))}
say('target 1317 response norms',target)
resp={k:[] for k in neurons};shifts={}
calib=fit_docs[:8]
for k in neurons:
    sub=Subject(model,xindex=k,mindex=1423)
    try:
        base={d['id']:sub.forward(d) for d in fit_docs}
        def mean_norm(A):
            v=[]
            for d in calib:
                s0=base[d['id']];dn=base[d['donor']];sg=np.sign(float(dn['X'])-float(s0['X'])) or 1.
                v.append(np.linalg.norm(sub.forward(d,x=float(s0['X'])+sg*A)['y'].astype(float)-s0['y'].astype(float)))
            return float(np.mean(v))
        for dose in (.5,1.):
            lo,hi=0.,1.
            while mean_norm(hi)<target[dose] and hi<1e4:hi*=2
            for _ in range(20):
                mid=(lo+hi)/2
                if mean_norm(mid)<target[dose]:lo=mid
                else:hi=mid
            shifts[(k,dose)]=hi;say('neuron',k,'dose',dose,'shift',hi,'calib norm',mean_norm(hi))
        for d in fit_docs:
            s0=base[d['id']];dn=base[d['donor']];sg=np.sign(float(dn['X'])-float(s0['X'])) or 1.
            for dose in (.5,1.):
                t=sub.forward(d,x=float(s0['X'])+sg*shifts[(k,dose)]);resp[k].append((t['y']-s0['y']).astype(float).ravel())
    finally:sub.close()
    say('neuron',k,'responses',len(resp[k]),'mean norm',float(np.mean([np.linalg.norm(v) for v in resp[k]])))
def topk(M,k):return np.linalg.svd(M,full_matrices=False)[2][:k].T
spaces={'B5_U':U,'matched3_rank8':topk(np.stack([v for k in neurons for v in resp[k]]),8)}
for k in neurons:spaces[f'matched_n{k}_rank8']=topk(np.stack(resp[k]),8)
# 2) gradient ceiling from FIT native endpoints
def suffix(rec,ys,grad=False):
    h=decode(rec['H']);r=decode(rec['R']);v=torch.as_tensor(np.asarray(ys,np.float32),dtype=h.dtype)
    if grad:v.requires_grad_(True)
    hh=h.expand(len(v),-1,-1).clone();hh[:,-1]=r[0,-1]+v
    ctx=torch.enable_grad() if grad else torch.inference_mode()
    with ctx:
        for L in range(22,28):hh=model.model.layers[L](hh,**decode(rec['kwargs'][str(L)]))
        z=torch.nn.functional.linear(model.model.norm(hh)[:,-1:],W);m=z[:,0,1]-z[:,0,0]
        if grad:m.sum().backward();return m.detach().double().numpy(),v.grad.double().numpy()
    return m.double().numpy(),None
R20={}
# 3) TEST evaluation
rows=[];t0=time.time()
for i,((did,b,dz),rec) in enumerate(sorted(tgt.items())):
    s=st(nat[did]);t=st(rec);D=(t['y']-s['y']).astype(float).ravel()
    P=decode(fc[f'{did}_{b}_{dz}']['old']).numpy().astype(float).ravel();y0=s['y'].astype(float).ravel();e=D-P
    truth=suffix(rec,[t['y'].ravel()])[0][0];mp,g=suffix(rec,[y0+P],grad=True);g=g[0]
    cand={};row=dict(id=did,block=docs[did]['block'],branch=b,dose=dz,truth=float(truth),P=float(mp[0]))
    for k,Q in spaces.items():
        cand['withP_'+k]=y0+P+Q@(Q.T@e);cand['noP_'+k]=y0+Q@(Q.T@D)
        row['g_in_'+k]=float(np.sum((Q.T@g)**2)/np.sum(g**2))
    for k,Q in R20.items():cand['withP_'+k]=y0+P+Q@(Q.T@e)
    names=list(cand);ms=[]
    for j in range(0,len(names),16):ms+=list(suffix(rec,[cand[n] for n in names[j:j+16]])[0])
    row.update({'m_'+n:float(v) for n,v in zip(names,ms)})
    gp=g-U@(U.T@g);ep=e-U@(U.T@e)
    row['cos_gperp_eperp']=float(gp@ep/(np.linalg.norm(gp)*np.linalg.norm(ep)+1e-30))
    rows.append(row)
    if i%64==0:say(i,len(tgt),f'{time.time()-t0:.0f}s')
json.dump(rows,open(OUT/'ctrl3_rows.json','w'))
blk=collections.defaultdict(list)
for r in rows:blk[r['block']].append(r)
def bmse(key):return float(np.mean([np.mean([(r[key]-r['truth'])**2 for r in v]) for v in blk.values()]))
base=bmse('P')
zero_ref=float(np.mean([np.mean([r['truth']**2 for r in v]) for v in blk.values()]))
res=dict(conditions=len(rows),blocks=len(blk),neurons=neurons,MSE_P=base,
 withP={k:dict(MSE=bmse('m_withP_'+k),reduction_pct=100*(1-bmse('m_withP_'+k)/base)) for k in spaces},
 noP={k:dict(MSE=bmse('m_noP_'+k)) for k in spaces},
 random20_withP_reduction_pct=[100*(1-bmse('m_withP_'+k)/base) for k in R20],
 grad_share={k:float(np.mean([r['g_in_'+k] for r in rows])) for k in spaces},
 mean_abs_cos_gperp_eperp=float(np.mean([abs(r['cos_gperp_eperp']) for r in rows])),reference_1_over_sqrt=1/np.sqrt(dim-8))
json.dump(res,open(OUT/'ctrl3_summary.json','w'),indent=1);say(json.dumps(res,indent=1));say('done',time.ctime())
