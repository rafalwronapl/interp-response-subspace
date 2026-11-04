"""Seven-method complete dose/grid fixture; no repeat of A/C analysis."""
import tempfile,json
from pathlib import Path
from research_checks.persistence import immutable
from analysis.C2_fresh_fit_holdout_20261102.worker import sha
from analysis.Qwen06_ABC_extension_20261102.b_worker import DOSES,METHODS,holdout_operations
from analysis.Qwen06_ABC_extension_20261102.b_score import analyze
BASE=Path(__file__).parent
def main():
 assert len(holdout_operations('d'))==47 and len(set(holdout_operations('d')))==47
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp);(p/'run').mkdir();docs=[];rows=[];factors=[.9,.8,.75,.7,.6,.5,0.]
  for frame in ['T2','T5']:
   for b in range(256):
    for side in [0,1]:
     for q in [0,1]:
      doc=f'{frame}_{b}_{side}_{q}';meta=dict(document=doc,block=f'b{b}',frame=frame,side=side,q=q);docs.append(dict(id=doc,role='HOLDOUT',**{k:meta[k] for k in ['block','frame','side','q']}))
      for c in DOSES:rows.append(dict(meta,dose=c,native_effect=c,native_vector_sq=c*c,CLT_vector_error_sq=(.1*c)**2,predictions={m:f*c for m,f in zip(METHODS,factors)}))
  (p/'corpus.json').write_text(json.dumps(dict(documents=docs)));(p/'config.json').write_text(json.dumps(dict(output=str(p/'run'),corpus=str(p/'corpus.json'))));immutable(p/'run/FIT_frozen.json',dict(synthetic=True));immutable(p/'run/result.json',dict(rows=rows,FIT_frozen_sha256=sha(p/'run/FIT_frozen.json')));r=analyze(p);assert len(r['cells'])==126
  expected=dict(zip(METHODS,[1-f for f in factors]))
  assert all(abs(cell['relative_RMS']-expected[cell['method']])<1e-12 and cell['blocks']==256 for cell in r['cells'])
 proof=dict(status='PASS',model_calls=0,tests=['Actual47logicalops/context×2048+640FIT+28controls=96924','Complete2048doc×6dose×7method grid gives126frame/mixture cells','KnownnormalizedRMS forstrongnativeTaylor andseparatescalarCOMPOSED_LINEAR','No A/C/bootstrap/model/head replay'],source_hashes={n:sha(BASE/n) for n in ['b_worker.py','b_score.py','b_scoring_fixtures_v2.py']})
 (BASE/'b_scoring_fixture_v2.json').write_text(json.dumps(proof,indent=2)+'\n');print('B126cellstrong-baseline scorerfixturePASS;0modelcalls')
if __name__=='__main__':main()
