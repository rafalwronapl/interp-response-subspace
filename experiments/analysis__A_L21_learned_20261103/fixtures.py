"""Synthetic regression, rank-deficient prior identity and fold separation."""
import numpy as np
from research_checks.persistence import immutable
from .common import BASE
from .models import fit,predict,folds
def run():
    rng=np.random.default_rng(3110361);x=rng.normal(size=(6,12));x-=x.mean(0);b=rng.normal(size=(12,8));y=rng.normal(size=(6,8));y-=y.mean(0)
    lam=.01;centered=b+x.T@np.linalg.solve(x@x.T+len(x)*lam*np.eye(len(x)),y-x@b)
    direct=np.linalg.solve(x.T@x+len(x)*lam*np.eye(x.shape[1]),x.T@y+len(x)*lam*b)
    np.testing.assert_allclose(centered,direct,atol=1e-10,rtol=1e-10)
    m=fit(x,y-x@b,dict(kernel='linear',lam=lam));np.testing.assert_allclose(x@b+predict(m,x),x@centered,atol=1e-10)
    small=rng.normal(size=(12,3));target=rng.normal(size=(12,2));prior=rng.normal(size=(3,2))
    a=prior+np.linalg.lstsq(small,target-small@prior,rcond=None)[0];d=np.linalg.lstsq(small,target,rcond=None)[0];np.testing.assert_allclose(a,d,atol=1e-10)
    # Unlike full-column-rank OLS, underdetermined min-norm direct can discard prior nullspace.
    a=b+np.linalg.lstsq(x,y-x@b,rcond=None)[0];d=np.linalg.lstsq(x,y,rcond=None)[0];assert np.linalg.norm(a-d)>1.
    blocks=np.repeat(np.arange(16),16);seen=np.zeros(len(blocks),int)
    for v in folds(blocks):
        assert not set(blocks[v])&set(blocks[~v]);seen+=v
    assert np.all(seen==1)
    for kernel in ['linear','rbf']:
        m=fit(np.ones((8,3)),np.ones((8,2)),dict(kernel=kernel,lam=.01));assert np.isfinite(predict(m,np.ones((2,3)))).all()
    immutable(BASE/'fixtures.json',dict(passed=True,rank_deficient_prior_identity=True,full_rank_zero_penalty_equivalence=True,
        naive_minimum_norm_equivalence_rejected=True,block_fold_separation=True,constant_kernel_finite=True,native_calls=0))
    print('synthetic checks PASS')
if __name__=='__main__':run()
