"""Focused precision, feature boundary and coupled-cache tests; no model."""
import tempfile,json,numpy as np
from pathlib import Path
from analysis.Qwen06_ABC_extension_20261102.evaluate import vector_hash,noD_predict,extract_noD_features,expected_arms,sha
from ab_confirmatory_20261101.runtime import Runtime
BASE=Path(__file__).parent
def main():
 rng=np.random.default_rng(2026110206);d=8;fit=dict(a=rng.normal(size=(4,d)),dual=rng.normal(size=(4,d)),zm=np.zeros(d),ym=np.zeros(d),scale=1.,regularization=.01,n=4);ps={i:fit for i in ['r1','r2','r3']};maps={(i,j):fit for i in ps for j in ps};us={i:np.linalg.qr(rng.normal(size=(d,2)))[0] for i in ps};r={k:rng.normal(size=d) for k in ['R','x','y']};donor=dict(x=rng.normal(size=d),y=object(),margin=object());assets=(dict(eps=1e-6,softcap=None),rng.normal(size=d),rng.normal(size=(2,d)),False)
 class Poison(dict):
  def __getitem__(self,k):
   assert k=='x','Forbidden target-field read'
   return super().__getitem__(k)
 a=noD_predict(extract_noD_features(r,donor),ps,maps,maps,us,assets);b=noD_predict(extract_noD_features(r,Poison(donor)),ps,maps,maps,us,assets);assert vector_hash(a[0])==vector_hash(b[0])
 assert vector_hash(dict(x=np.arange(6,dtype=np.float64)))!=vector_hash(dict(x=np.arange(6,dtype=np.float32)))
 assert vector_hash(a[0])==vector_hash(dict(reversed(list(a[0].items()))))
 with tempfile.TemporaryDirectory() as tmp:
  path=Path(tmp);arms=expected_arms();assert len(arms)==50;schedule=dict(operations=[dict(id=arm,kind='forward') for arm in arms]);identity=dict(model='synthetic',code='fixture');calls=[]
  def compute():calls.append(1);return {arm:float(k) for k,arm in enumerate(arms)}
  rt=Runtime(path,identity,schedule);kwargs=dict(document=dict(id='fixture'),intervention=dict(arms=arms),readout=dict(answer_ids=[374,525]));first=rt.compute(arms[0],**kwargs,function=compute,extra_operations=arms[1:]);assert len(list((path/'cache').glob('*.json')))==1 and len(list((path/'completed').glob('*.json')))==50
  hashes={str(p):sha(p) for p in (path/'completed').glob('*.json')};rt=Runtime(path,identity,schedule);second=rt.compute(arms[0],**kwargs,function=compute,extra_operations=arms[1:]);assert first==second and len(calls)==1 and all(sha(Path(p))==h for p,h in hashes.items());assert len(rt.attempts)==50
 result=dict(status='PASS',pretrained_calls=0,tests=['Exact noD poison donor target boundary','Reproducible dtype/shape/value-sensitive FP64 vectorhash and order invariance','Coupled50logicalF one durable cache and50immutable receipts','Resume performszeroextraheadcomputations andpreservesbytes'],source_hashes={n:sha(BASE/n) for n in ['evaluate.py','evaluation_fixtures.py','score.py']})
 (BASE/'evaluation_fixture_v1.json').write_text(json.dumps(result,indent=2)+'\n');print('QBase evaluation focusedfixtures PASS,0model/headcalls')
if __name__=='__main__':main()
