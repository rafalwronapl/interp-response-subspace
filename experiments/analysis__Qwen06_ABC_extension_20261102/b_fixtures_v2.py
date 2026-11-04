"""Synthetic SwiGLU derivative, gate and dose operator fixtures only."""
import json,struct,numpy as np,torch
from pathlib import Path
from analysis.Qwen06_ABC_extension_20261102.b_math import branch,branch_jvp,clt_delta,clt_response
from analysis.C2_fresh_fit_holdout_20261102.worker import ROOT,sha
BASE=Path(__file__).parent
def main():
 torch.set_num_threads(1);torch.manual_seed(2026110206);d=8;h=12;k=17;a=dict(native={n:torch.randn(shape,dtype=torch.float64)*.2 for n,shape in [('gate',(h,d)),('up',(h,d)),('down',(d,h))]},encoder=torch.randn(k,d,dtype=torch.float64),bias=torch.randn(k,dtype=torch.float64),threshold=torch.rand(k,dtype=torch.float64)+.1,decoder=torch.randn(k,d,dtype=torch.float64));x=torch.randn(d,dtype=torch.float64);dx=torch.randn(d,dtype=torch.float64)
 exact=branch_jvp(x,dx,a);eps=1e-5;numeric=(branch(x+eps*dx,a)-branch(x-eps*dx,a))/(2*eps);assert torch.allclose(exact,numeric,atol=1e-9,rtol=1e-7)
 zero,frozen=clt_delta(x,x,a);assert torch.count_nonzero(zero)==0 and torch.count_nonzero(frozen)==0
 # For a fixed activation mask, difference prediction is exactly its affine
 # derivative; threshold crossings are intentionally allowed in the full CLT.
 pc=torch.nn.functional.linear(x+dx,a['encoder'],a['bias']);p0=torch.nn.functional.linear(x,a['encoder'],a['bias']);mask=p0>a['threshold'];_,f=clt_delta(x,x+dx,a);assert torch.allclose(f,torch.nn.functional.linear((pc-p0)*mask,a['decoder'].T),atol=1e-12,rtol=0)
 tc,fr=clt_response(x,x+dx,dx,a);oldtc,oldfr=clt_delta(x,x+dx,a);assert torch.equal(tc,oldtc) and torch.allclose(fr,oldfr,atol=1e-12,rtol=0)
 zero,_=clt_response(x,x.repeat(6,1),torch.zeros_like(dx),a);assert torch.count_nonzero(zero)==0
 batch,_=clt_response(x,torch.stack([x+.05*dx,x+dx,x+2*dx]),dx,a);assert torch.equal(batch[1],tc)
 headers={}
 for name in ['clt_cache/e017/W_enc_27.safetensors','clt_cache/e033/W_dec_27.safetensors']:
  with (ROOT/name).open('rb') as stream:n=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(n))
  headers[name]={k:v for k,v in header.items() if k!='__metadata__'}
 out=dict(status='PASS',pretrained_model_calls=0,synthetic_calls_only=True,tests=['SwiGLU directional derivative versus symmetric FP64 finite difference','CLTzero-delta exactzero and frozenmask formula'],actual_tensor_headers=headers,source_hashes={n:sha(BASE/n) for n in ['b_math.py','b_fixtures_v2.py']})
 (BASE/'b_fixture_v2.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(headers,indent=2))
if __name__=='__main__':main()
