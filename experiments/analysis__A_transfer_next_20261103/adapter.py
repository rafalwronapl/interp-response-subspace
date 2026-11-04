"""Exact existing suffix with four head rows; both readouts share computation."""
import numpy as np,torch
from analysis.AB_five_experiments_20261103.A_L21.adapter_v4 import Adapter as Original
from ab_confirmatory_20261101.native_tail_b import decode
class Adapter(Original):
 def __init__(self,model,heads):
  super().__init__(model);self.heads=heads;self.four_weight=model.lm_head.weight[[heads[k] for k in [' is',' are',' has',' have']]].detach().clone()
 def both(self,record,ys):
  h=decode(record['H']);r=decode(record['R']);v=torch.as_tensor(np.asarray(ys,np.float32),dtype=h.dtype)
  if v.ndim!=2 or v.shape[1]!=h.shape[-1] or not 1<=len(v)<=16:raise ValueError('Bad suffix batch')
  h=h.expand(len(v),-1,-1).clone();h[:,-1]=r[0,-1]+v
  with torch.inference_mode():
   for layer in range(22,28):h=self.model.model.layers[layer](h,**decode(record['kwargs'][str(layer)]))
   z=torch.nn.functional.linear(self.model.model.norm(h)[:,-1:],self.four_weight)[:,0]
  return torch.stack([z[:,1]-z[:,0],z[:,3]-z[:,2]],1).double().numpy().tolist()
 def full_both(self,doc,x,m,Y):
  captured=[]
  def hook(module,args,out):captured.append(out[0,-1,[self.heads[k] for k in [' is',' are',' has',' have']]].detach().double().numpy())
  handle=self.model.lm_head.register_forward_hook(hook)
  try:self.full(doc,x,m,Y)
  finally:handle.remove()
  z=captured[0];return [float(z[1]-z[0]),float(z[3]-z[2])]
