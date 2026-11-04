"""Descriptive paired-block A/C analysis; do not select fitting groups."""
import json,numpy as np
from pathlib import Path
from analysis.C2_fresh_fit_holdout_20261102.worker import read,receipt,sha,plain
from research_checks.persistence import immutable
from analysis.Qwen06_ABC_extension_20261102.evaluate import expected_arms
def analyze(stage):
 root=Path(__file__).resolve().parents[2];stage=Path(stage);cfg=read(stage/'config.json');out=root/cfg['output'];raw=receipt(out/'result.json');fr=receipt(out/'noD_forecast.json');ore=receipt(out/'oracle_forecast.json');forecast=fr['records'];oracle=ore['records'];bydoc={}
 assert cfg['model']=='qwen06_base' and cfg['arms']==expected_arms()
 corpus=read(root/cfg['corpus'])['documents'];assert len({d['id'] for d in corpus})==len(corpus);hold={d['id']:d for d in corpus if d['role']=='HOLDOUT'};assert len(hold)==2048
 assert raw['noD_forecast_sha256']==ore['noD_forecast_sha256']==sha(out/'noD_forecast.json') and raw['oracle_forecast_sha256']==sha(out/'oracle_forecast.json')
 for r in raw['rows']:
  doc=r['document'];assert doc in hold and r['arm'] in cfg['arms'] and r['arm'] not in bydoc.get(doc,{}) and np.isfinite(r['margin']);assert all(r[k]==hold[doc][k] for k in ['block','frame','side','q']);bydoc.setdefault(doc,{})[r['arm']]=r['margin']
 assert set(bydoc)==set(hold) and all(set(v)==set(cfg['arms']) for v in bydoc.values())
 grid={(d,i,j) for d in hold for i in ['r1','r2','r3'] for j in ['r1','r2','r3']}
 for records,kind in [(forecast,'noD'),(oracle,'oracle')]:
  seen=set()
  for r in records:
   key=r['document'],r['i'],r['j'];assert key in grid and key not in seen;seen.add(key);assert all(r[k]==hold[r['document']][k] for k in ['block','frame','side','q'])
   if kind=='noD':
    assert np.isfinite(r['score_projected']) and np.isfinite(r['score_full']) and r['score_full']<=0
    assert type(r['apply_projected']) is bool and r['apply_projected']==(r['score_projected']<0) and type(r['apply_full']) is bool and r['apply_full']==(r['score_full']<0)
   else:assert all(np.isfinite(r[k]) for k in ['score','energy_P','energy_inside','energy_outside']) and all(r[k]>=0 for k in ['energy_P','energy_inside','energy_outside'])
  assert seen==grid
 f={(r['document'],r['i'],r['j']):r for r in forecast};o={(r['document'],r['i'],r['j']):r for r in oracle};meta={r['document']:r for r in raw['rows']};groups=['r1','r2','r3'];summary={};blockrecords=[]
 for frame in ['T5','T2']:
  for scope,pairs in [('primary_matched_three',[(i,i) for i in groups]),('secondary_all_nine',[(i,j) for i in groups for j in groups])]:
   rows=[]
   for doc,m in bydoc.items():
    d=meta[doc]
    if d['frame']!=frame:continue
    def loss(arm):
     value=(m[arm]-m['native'])**2;assert np.isfinite(value);return value
    for i,j in pairs:
     rec=f[doc,i,j];oref=o[doc,i,j];P=loss('P_'+i);inside=loss('oracle_'+i+'_'+j);proj=loss('proj_'+i+'_'+j);outside=loss('outside_'+i+'_'+j)
     rows.append(dict(block=d['block'],i=i,j=j,P=P,inside=inside,outside=outside,projected=proj,full=loss('full_'+i+'_'+j),direct=loss('direct_'+i+'_'+j),gated=proj if rec['apply_projected'] else P,energy_inside=oref['energy_inside'],energy_outside=oref['energy_outside'],delta_inside_P=inside-P,delta_inside_outside=inside-outside,oracle_correct=float(np.sign(oref['score'])==np.sign(inside-P)),noD_correct=float(np.sign(rec['score_projected'])==np.sign(proj-P)),help=float(proj<P),harm=float(proj>P)))
     rows[-1].update(oracle_nonzero=float(oref['score']!=0),noD_nonzero=float(rec['score_projected']!=0),oracle_correct_nonzero=float(oref['score']!=0 and np.sign(oref['score'])==np.sign(inside-P)),noD_correct_nonzero=float(rec['score_projected']!=0 and np.sign(rec['score_projected'])==np.sign(proj-P)),oracle_actual_tie=float(inside==P),noD_actual_tie=float(proj==P),apply_fraction=float(rec['apply_projected']))
   keys=[k for k in rows[0] if k not in ['block','i','j']];blocks=sorted({r['block'] for r in rows});assert len(blocks)==256
   grouped={b:[] for b in blocks}
   for row in rows:grouped[row['block']].append([row[k] for k in keys])
   matrix=np.array([np.mean(grouped[b],axis=0) for b in blocks])
   summary[frame+'_'+scope]=dict(blocks=256,dependent_records=len(rows),means=dict(zip(keys,map(float,matrix.mean(0)))),per_pair={i+'_'+j:{k:float(np.mean([r[k] for r in rows if r['i']==i and r['j']==j])) for k in keys} for i,j in pairs})
   means=summary[frame+'_'+scope]['means'];summary[frame+'_'+scope]['sign_metrics']=dict(exact_ties_compatible_accuracy_oracle=means['oracle_correct'],exact_ties_compatible_accuracy_noD=means['noD_correct'],coverage_oracle=means['oracle_nonzero'],coverage_noD=means['noD_nonzero'],conditional_accuracy_oracle=means['oracle_correct_nonzero']/means['oracle_nonzero'] if means['oracle_nonzero'] else None,conditional_accuracy_noD=means['noD_correct_nonzero']/means['noD_nonzero'] if means['noD_nonzero'] else None,always_help_accuracy=means['help'],always_harm_accuracy=means['harm'],zero_score_policy='Abstain; gate skips. Exact-ties-compatible accuracy retained for comparison with earlierC3, separately from nonzero conditional accuracy.')
   blockrecords.extend(dict(frame=frame,scope=scope,block=b,**dict(zip(keys,map(float,values)))) for b,values in zip(blocks,matrix))
 # ONE common A analysis across models, with no second bootstrap of C metrics.
 from analysis.A_gemma270_extension_20261102.analyze import analyze as aligned_analyze
 names={'native':'native'}
 for i in groups:names.update({'P_'+i:'P_'+i,'oracle_'+i+'_'+i:'inside_'+i,'outside_'+i+'_'+i:'outside_'+i})
 ar=[dict(row,arm=names[row['arm']]) for row in raw['rows'] if row['arm'] in names]
 ae=[dict(document=r['document'],block=r['block'],frame=r['frame'],side=r['side'],q=r['q'],fit=r['i'],P=r['energy_P'],inside=r['energy_outside'],outside=r['energy_inside']) for r in oracle if r['i']==r['j']]
 acorpus=dict(documents=[dict(id=doc,**{k:meta[doc][k] for k in ['block','frame','side','q']}) for doc in bydoc])
 aligned=aligned_analyze(dict(rows=ar,energies=ae),acorpus)
 result=dict(model='Qwen3-0.6B-Base',layer=27,primary='T5 primary_matched_three delta_inside_P',scope='FixedFIT256sharedblocks. Common A frozen1999draw bootstrap-t95 descriptive intervals only; C metrics descriptive means, no duplicate resampling, no qualifiedFWER/populationcoverage. HistoricalE041L21differentoperator.',aligned_A=aligned,summary=summary,source_sha256=sha(out/'result.json'))
 immutable(stage/'summary.json',plain(result));immutable(stage/'block_summary.json',plain(blockrecords));return result
if __name__=='__main__':
 import sys
 print(json.dumps(analyze(sys.argv[1]),indent=2))
