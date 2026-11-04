"""Complete descriptive dose table from accepted bounded Qwen CLT results."""
import sys,json,numpy as np
from pathlib import Path
from analysis.C2_fresh_fit_holdout_20261102.worker import ROOT,read,receipt,sha,plain
from research_checks.persistence import immutable
from analysis.Qwen06_ABC_extension_20261102.b_worker import DOSES,METHODS
def analyze(stage):
 stage=Path(stage);cfg=read(stage/'config.json');source=ROOT/cfg['output']/'result.json';raw=receipt(source);rows=raw['rows'];assert len(rows)==12288 and raw['FIT_frozen_sha256']==sha(ROOT/cfg['output']/'FIT_frozen.json')
 hold={d['id']:d for d in read(ROOT/cfg['corpus'])['documents'] if d['role']=='HOLDOUT'};assert len(hold)==2048;seen=set()
 for r in rows:
  key=r['document'],r['dose'];assert r['document'] in hold and r['dose'] in DOSES and key not in seen;seen.add(key);assert all(r[k]==hold[r['document']][k] for k in ['block','frame','side','q']);assert set(r['predictions'])==set(METHODS) and all(np.isfinite(v) for v in r['predictions'].values());assert all(np.isfinite(r[k]) for k in ['native_effect','native_vector_sq','CLT_vector_error_sq']) and r['native_vector_sq']>=0 and r['CLT_vector_error_sq']>=0
 assert seen=={(doc,c) for doc in hold for c in DOSES}
 cells=[];blocks=[]
 for frame in ['T2','T5','mixture']:
  for dose in DOSES:
   selected=[r for r in rows if r['dose']==dose and (frame=='mixture' or r['frame']==frame)];assert len(selected)==(2048 if frame=='mixture' else 1024);native=np.array([r['native_effect'] for r in selected]);native_mse=float(np.mean(native*native))
   for method in METHODS:
    errors=np.array([r['predictions'][method]-r['native_effect'] for r in selected]);mse=float(np.mean(errors*errors));cells.append(dict(frame=frame,dose=dose,method=method,contexts=len(selected),blocks=256,native_effect_RMS=float(np.sqrt(native_mse)),absolute_error_RMS=float(np.sqrt(mse)),relative_RMS=float(np.sqrt(mse/native_mse)) if native_mse>0 else None,relative_RMS_defined=bool(native_mse>0),signed_error=float(np.mean(errors))))
    for block in sorted({r['block'] for r in selected}):
     ix=[i for i,r in enumerate(selected) if r['block']==block];blocks.append(dict(frame=frame,dose=dose,method=method,block=block,native_MSE=float(np.mean(native[ix]**2)),error_MSE=float(np.mean(errors[ix]**2))))
 result=dict(model='Qwen3-0.6B-Base',layer=27,public_artifact='bluelightai/clt-qwen3-0.6b-base-20k source27target27',operator='full normalizedinput interpolation',readout='is-minus-are',source_sha256=sha(source),scope='Complete descriptive126celltable,256sharedblocks,conceptualcrossartifactextension; no qualifiedCI/FWER and no identicalGemmaBreplication claim. MainTAYLOR is nativeMLPvectorJVP propagatedthroughactualnonlinearhead, COMPOSED_LINEAR is separateweakerdiagnostic.',cells=cells)
 immutable(stage/'summary.json',plain(result));immutable(stage/'block_summary.json',plain(blocks));return result
if __name__=='__main__':print(json.dumps(analyze(sys.argv[1]),indent=2))
