"""Synthetic inside/outside algebra and residual-energy invariants; no model."""
import json
import numpy as np
from pathlib import Path
from analysis.A_gemma270_extension_20261102.evaluate import updates,FITS
from analysis.C2_fresh_fit_holdout_20261102.worker import sha,put

def main():
 rng=np.random.default_rng(73);n=12;rank=3;q,_=np.linalg.qr(rng.normal(size=(n,rank)));D=rng.normal(size=n);p=rng.normal(size=n)
 # Compact ridge object with constant prediction; same frozen predictor path.
 fit=dict(a=np.zeros((1,n)),dual=np.zeros((1,n)),zm=np.zeros(n),ym=p,scale=1.,regularization=.01,n=1)
 rec=dict(y=np.zeros(n),x=np.zeros(n));donor=dict(y=D,x=np.ones(n))
 values,energy=updates(rec,donor,{i:fit for i in FITS},{i:q for i in FITS})
 for i,e in zip(FITS,energy):
  inside=values['inside_'+i];outside=values['outside_'+i]
  assert np.allclose(inside+outside,D+p)
  assert np.allclose(e['P'],e['inside']+e['outside'])
  assert np.allclose(np.sum((D-inside)**2),e['inside']) and np.allclose(np.sum((D-outside)**2),e['outside'])
 base=Path(__file__).parent;put(base/'head_fixture_v5.json',dict(status='PASS',tests=['inside_outside_complement','orthogonal_residual_energy_decomposition','named_arm_residual_norms'],model_calls=0,source_sha256=sha(base/'head_fixture_v1.py'),evaluate_sha256=sha(base/'evaluate.py')))
if __name__=='__main__':main()
