"""Cheap design/math/accounting fixtures; no pretrained model."""
from pathlib import Path
import json,numpy as np,tempfile
from .math import fit,predict,corrections,RANKS
from .worker import schedule,arms,score
from analysis.Qwen06_ABC_extension_20261102.bundle_runtime import BundleRuntime
BASE=Path(__file__).parent
def main():
 rng=np.random.default_rng(88);x=rng.normal(size=(128,40));z=rng.normal(size=(128,40));D=rng.normal(size=(128,40))
 f=fit(x,z,D,3110340)
 for rank in RANKS:
  U=f['U'][:,:rank];V=f['random_U'][:,:rank]
  np.testing.assert_allclose(U.T@U,np.eye(rank),atol=1e-12)
  ca,cb,cc=corrections(D[0]-predict(f,x[0],z[0]),U,V)
  np.testing.assert_allclose(np.linalg.norm(ca),np.linalg.norm(cc),atol=1e-12)
 docs=json.loads((BASE/'corpus.json').read_text())['documents'];ops=schedule(docs)
 assert len(ops['operations'])==66656 and len({o['id'] for o in ops['operations']})==66656
 assert len(arms())==62
 rows=[dict(block=b,truth=float(b),branch=branch,dose=dose,margins={a:float(b)+i*.01 for i,a in enumerate(arms())}) for b in range(64) for branch in ['B','C'] for dose in [.5,1.] for _ in range(4)]
 report=score(rows);assert report['conditions']==1024 and len(report['rank_summary'])==18
 with tempfile.TemporaryDirectory() as tmp:
  ident=dict(model='test',code='test');sch=dict(operations=[dict(id='a',kind='forward'),dict(id='b',kind='forward')])
  rt=BundleRuntime(tmp,ident,sch);v=rt.compute('a',dict(id=1),dict(targets=[1,2]),dict(ids=[525,374]),lambda:dict(a=1.,b=2.),extra_operations=['b'])
  out=BundleRuntime(tmp,ident,sch).compute('a',dict(id=1),dict(targets=[1,2]),dict(ids=[525,374]),lambda:(_ for _ in ()).throw(Exception('recompute')),extra_operations=['b'])
  assert out==v
 result=dict(hooks=True,data=True,pilot=True,accounting=True,restart=True,max_forward=66656,backward=0,trained_calls=0,scope='math/projection/64blockanalysis/schedule/bundle-resume; pretrainedqualification separate')
 (BASE/'main_fixtures.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
