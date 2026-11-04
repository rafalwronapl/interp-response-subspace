"""Frozen scalar-only A extension analysis; block, not fit, is the unit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from ab_confirmatory_20261101.statistics import mean_contrast

FITS = ('r1', 'r2', 'r3')
ARMS = {'native'} | {f'{a}_{f}' for a in ('P','inside','outside') for f in FITS}

def load(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if 'payload' in obj:
        from research_checks.gate import digest
        if digest(obj['payload']) != obj['payload_sha256']:
            raise ValueError('Invalid immutable payload')
        return obj['payload']
    return obj

def analyze(result, corpus):
    docs = {d['id']: d for d in corpus['documents']}
    assert len(docs) == len(corpus['documents']) == 2048
    blocks = sorted({d['block'] for d in docs.values()})
    assert len(blocks) == 256
    for b in blocks:
        assert {(d['frame'], d['side'], d['q']) for d in docs.values() if d['block']==b} == {
            (f,s,q) for f in ('T2','T5') for s in (0,1) for q in (0,1)}
    values = {}
    for row in result['rows']:
        doc, arm = row['document'], row['arm']
        assert doc in docs and arm in ARMS and (doc,arm) not in values
        assert all(row[k] == docs[doc][k] for k in ('block','frame','side','q'))
        assert np.isfinite(row['margin'])
        values[doc,arm] = float(row['margin'])
    assert set(values) == {(d,a) for d in docs for a in ARMS}
    energies = {}
    for e in result['energies']:
        doc, fit = e['document'], e['fit']
        assert doc in docs and fit in FITS and (doc,fit) not in energies
        assert all(e[k] == docs[doc][k] for k in ('block','frame','side','q'))
        assert all(np.isfinite(e[k]) and e[k]>=0 for k in ('P','inside','outside'))
        energies[doc,fit] = {k:float(e[k]) for k in ('P','inside','outside')}
    assert set(energies) == {(d,f) for d in docs for f in FITS}
    records = []
    for doc,d in docs.items():
        for fit in FITS:
            losses = {a:(values[doc,f'{a}_{fit}']-values[doc,'native'])**2 for a in ('P','inside','outside')}
            assert all(np.isfinite(v) for v in losses.values())
            records.append(dict(document=doc,fit=fit,**{k:d[k] for k in ('block','frame','side','q')},
                losses=losses,vector_energies=energies[doc,fit]))
    frames = {}
    for frame in ('T5','T2'):
        rows = [r for r in records if r['frame']==frame]
        byblock = []
        for b in blocks:
            rr = [r for r in rows if r['block']==b]
            assert len(rr)==12
            means = {a:float(np.mean([r['losses'][a] for r in rr])) for a in ('P','inside','outside')}
            byblock.append(dict(block=b,losses=means,inside_minus_P=means['inside']-means['P'],
                inside_minus_outside=means['inside']-means['outside']))
        contrasts = {}
        for idx,key in enumerate(('inside_minus_P','inside_minus_outside')):
            seed=2026110270+(0 if frame=='T5' else 2)+idx
            stat=mean_contrast([b[key] for b in byblock],draws=1999,seed=seed,alpha=.05)
            contrasts[key]={k:v for k,v in stat.items() if k in ('status','reason','estimate','se','n','ci','draws','seed')}
        frames[frame]=dict(blocks=256,dependent_records=len(rows),
            losses={a:float(np.mean([b['losses'][a] for b in byblock])) for a in ('P','inside','outside')},
            contrasts=contrasts,per_block=byblock,
            per_fit={fit:dict(losses={a:float(np.mean([r['losses'][a] for r in rows if r['fit']==fit]))
                for a in ('P','inside','outside')},vector_energies={a:float(np.mean([
                    r['vector_energies'][a] for r in rows if r['fit']==fit])) for a in ('P','inside','outside')}) for fit in FITS})
    return dict(status='complete',frames=frames,records=records,
        primary='T5 inside_minus_P, four conditions and three fixed fits averaged within each of256blocks',
        scope='T2 and inside-minus-outside are descriptive. Bootstrap-t95% intervals use1999 fixed draws; no family decision, coverage/power guarantee or inference over refitted predictors. Positive loss difference means worse intervention-effect prediction, not worse natural-answer quality. Vector energies are squared residual norms of the named arms.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--corpus',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=analyze(load(a.result),load(a.corpus))
    out['source_sha256']={str(Path(v)):hashlib.sha256(Path(v).read_bytes()).hexdigest() for v in (a.result,a.corpus)}
    with Path(a.out).open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False)
