"""Literal original-Base last-MLP/public-CLT component math. No model loading."""
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from analysis.C2_fresh_fit_holdout_20261102.worker import ROOT,analytic_gradient

def load_assets():
 with safe_open(str(ROOT/'model/model.safetensors'),framework='pt',device='cpu') as f:
  native={kind:f.get_tensor('model.layers.27.mlp.'+kind+'_proj.weight').float() for kind in ['gate','up','down']}
 enc=load_file(str(ROOT/'clt_cache/e017/W_enc_27.safetensors'))
 dec=load_file(str(ROOT/'clt_cache/e033/W_dec_27.safetensors'))
 assets=dict(native=native,encoder=enc['W_enc_27'].float(),bias=enc['b_enc_27'].float(),threshold=enc['threshold_27'].float(),decoder=dec['W_dec_27'][:,0,:].float())
 assert assets['encoder'].shape==(20480,1024) and assets['decoder'].shape==(20480,1024)
 return assets

def branch(x,assets):
 w=assets['native'];return torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(x,w['gate']))*torch.nn.functional.linear(x,w['up']),w['down'])

def branch_jvp(x,dx,assets):
 w=assets['native'];g=torch.nn.functional.linear(x,w['gate']);u=torch.nn.functional.linear(x,w['up']);dg=torch.nn.functional.linear(dx,w['gate']);du=torch.nn.functional.linear(dx,w['up']);s=torch.sigmoid(g)
 return torch.nn.functional.linear((s*(1+g*(1-s))*dg)*u+torch.nn.functional.silu(g)*du,w['down'])

def clt_delta(x,xc,assets):
 p0=torch.nn.functional.linear(x,assets['encoder'],assets['bias']);pc=torch.nn.functional.linear(xc,assets['encoder'],assets['bias']);m0=p0>assets['threshold'];f0=p0*m0;fc=pc*(pc>assets['threshold'])
 return torch.nn.functional.linear(fc-f0,assets['decoder'].T),torch.nn.functional.linear((pc-p0)*m0,assets['decoder'].T)

def clt_response(x,xc,dx,assets):
 p0=torch.nn.functional.linear(x,assets['encoder'],assets['bias']);mask=p0>assets['threshold'];f0=p0*mask
 direction=torch.nn.functional.linear(dx,assets['encoder'])*mask
 # Same vector-shaped public-CLT kernel for baseline and every endpoint avoids
 # artificial threshold crossings from changing BLAS batch shape.
 def endpoint(row):
  pc=torch.nn.functional.linear(row,assets['encoder'],assets['bias']);fc=pc*(pc>assets['threshold']);return torch.nn.functional.linear(fc-f0,assets['decoder'].T)
 tc=endpoint(xc) if xc.ndim==1 else torch.stack([endpoint(row) for row in xc])
 return tc,torch.nn.functional.linear(direction,assets['decoder'].T)

def suffix(R,y,updates,norm,head,eps):
 z=R+(y.double()+updates.double()).float();z=z*torch.rsqrt(z.pow(2).mean(-1,keepdim=True)+eps);logits=torch.nn.functional.linear(z*norm,head);return (logits[...,0]-logits[...,1]).double().numpy()
