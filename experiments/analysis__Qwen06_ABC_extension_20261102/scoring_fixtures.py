"""Known-loss end-to-end masks for both new scorers, zero model calls."""
import tempfile,json
from pathlib import Path
from research_checks.persistence import immutable
from analysis.Qwen06_ABC_extension_20261102.evaluate import expected_arms,sha
from analysis.Qwen06_ABC_extension_20261102.score import analyze
from analysis.Qwen06_ABC_extension_20261102.b_score import analyze as analyze_b
from analysis.Qwen06_ABC_extension_20261102.b_worker import DOSES,METHODS
BASE=Path(__file__).parent
def main():
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp);(p/'run').mkdir();(p/'config.json').write_text(json.dumps(dict(output=str(p/'run'))));rows=[];fore=[];oracle=[];brows=[]
  for frame in ['T2','T5']:
   for block in range(256):
    for side in [0,1]:
     for q in [0,1]:
      doc=f'{frame}_{block}_{side}_{q}';meta=dict(document=doc,block=f'b{block}',frame=frame,side=side,q=q)
      for arm in expected_arms():
       value=0. if arm=='baseline' else 1. if arm=='native' else .5 if arm.startswith('P_') else .7 if arm.startswith('direct_') else .6 if arm.startswith('full_') else .8 if arm.startswith('proj_') else .9 if arm.startswith('oracle_') else .4
       rows.append(dict(meta,arm=arm,margin=value))
      for i in ['r1','r2','r3']:
       for j in ['r1','r2','r3']:
        fore.append(dict(meta,i=i,j=j,score_projected=-.1,apply_projected=True,score_full=-.16,apply_full=True));oracle.append(dict(meta,i=i,j=j,score=-.24,energy_P=5.,energy_inside=1.,energy_outside=4.))
      for c in DOSES:brows.append(dict(meta,dose=c,native_effect=c,native_vector_sq=c*c,CLT_vector_error_sq=(.1*c)**2,predictions={method:factor*c for method,factor in zip(METHODS,[.9,.8,.7,.6,.5,0.])}))
  docs={r['document']:dict(id=r['document'],role='HOLDOUT',**{k:r[k] for k in ['block','frame','side','q']}) for r in rows};(p/'corpus.json').write_text(json.dumps(dict(documents=list(docs.values()))));(p/'config.json').write_text(json.dumps(dict(output=str(p/'run'),model='qwen06_base',arms=expected_arms(),corpus=str(p/'corpus.json'))))
  immutable(p/'run/noD_forecast.json',dict(records=fore));nh=sha(p/'run/noD_forecast.json');immutable(p/'run/oracle_forecast.json',dict(records=oracle,noD_forecast_sha256=nh));immutable(p/'run/result.json',dict(rows=rows,noD_forecast_sha256=nh,oracle_forecast_sha256=sha(p/'run/oracle_forecast.json')));result=analyze(p)
  for frame in ['T2','T5']:
   r=result['summary'][frame+'_primary_matched_three'];assert r['blocks']==256 and r['dependent_records']==3072;assert abs(r['means']['delta_inside_P']+.24)<1e-12 and abs(r['means']['delta_inside_outside']+.35)<1e-12 and abs(r['means']['gated']-.04)<1e-12;assert r['means']['oracle_correct']==1 and r['means']['noD_correct']==1
  pb=p/'B';(pb/'run').mkdir(parents=True);(pb/'config.json').write_text(json.dumps(dict(output=str(pb/'run'),corpus=str(p/'corpus.json'))));immutable(pb/'run/FIT_frozen.json',dict(scope='synthetic'));immutable(pb/'run/result.json',dict(rows=brows,FIT_frozen_sha256=sha(pb/'run/FIT_frozen.json')));bresult=analyze_b(pb);assert len(bresult['cells'])==108
  for cell in bresult['cells']:
   expected=dict(zip(METHODS,[.1,.2,.3,.4,.5,1.]))[cell['method']];assert abs(cell['relative_RMS']-expected)<1e-12 and cell['blocks']==256
 receipt=dict(status='PASS',model_loads=0,pretrained_calls=0,tests=['A/C full2048contexts50arms knownloss means and3072dependentmatchedcells perframe','A/C matched/allnine frame masks andpointwiseblockintervals','B complete108cells dose/frame/mixture RMS normalizedratios'],source_hashes={n:sha(BASE/n) for n in ['score.py','b_score.py','scoring_fixtures.py']})
 (BASE/'scoring_fixture_v1.json').write_text(json.dumps(receipt,indent=2)+'\n');print('A/C+B end-to-end known-loss scoring fixtures PASS,0models')
if __name__=='__main__':main()
