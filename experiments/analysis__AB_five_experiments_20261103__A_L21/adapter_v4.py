"""Literal E041 neuron interventions with exact dynamic Qwen L22--27 reuse."""
import numpy as np
import torch
from e039.subject import Subject
from ab_confirmatory_20261101.native_tail_b import encode,decode
def enc(v):
 a=np.asarray(v)
 return str(a) if a.dtype.kind in 'US' else encode(torch.as_tensor(a))

class Adapter:
 def __init__(self,model):
  self.model=model;self.subject=Subject(model);self.collect=False;self.handles=[]
  if model.lm_head.bias is not None:raise ValueError('Expected bias-free Qwen head')
  self.readout_weight=model.lm_head.weight[[374,525]].detach().clone()
  def residual(module,args):
   if self.collect:self.R=encode(args[0])
  def boundary(module,args,out):
   if self.collect:self.H=encode(out)
  self.handles.append(model.model.layers[21].post_attention_layernorm.register_forward_pre_hook(residual))
  self.handles.append(model.model.layers[21].register_forward_hook(boundary))
  for layer in range(22,28):
   def kw(module,args,kwargs,layer=layer):
    if self.collect:
     if kwargs.get('past_key_values') is not None:raise ValueError('KV forbidden')
     self.kw[str(layer)]=encode(kwargs)
   self.handles.append(model.model.layers[layer].register_forward_pre_hook(kw,with_kwargs=True))
 def full(self,doc,x=None,m=None,replacement=None):
  self.collect=True;self.R=self.H=None;self.kw={}
  try:s=self.subject.forward(doc,x=x,m=m,replacement=replacement,poison=replacement is not None)
  finally:self.collect=False
  if self.R is None or self.H is None or len(self.kw)!=6:raise ValueError('Incomplete suffix capture')
  return dict(state={k:enc(v) for k,v in s.items()},R=self.R,H=self.H,kwargs=self.kw)
 def prefix(self,doc,x,m):
  s=self.subject.forward(doc,prefix=True,x=x,m=m)
  if 'Y' in s or 'y' in s or 'logits' in s:raise ValueError('Target leakage')
  return {k:enc(v) for k,v in s.items()}
 def margin_many(self,record,ys):
  h=decode(record['H']);r=decode(record['R']);values=torch.as_tensor(np.asarray(ys,np.float32),dtype=h.dtype)
  if values.ndim!=2 or values.shape[1]!=h.shape[-1] or not 1<=len(values)<=16:raise ValueError('Bad patch batch')
  h=h.expand(len(values),-1,-1).clone();h[:,-1]=r[0,-1]+values
  with torch.inference_mode():
   for layer in range(22,28):h=self.model.model.layers[layer](h,**decode(record['kwargs'][str(layer)]))
   z=torch.nn.functional.linear(self.model.model.norm(h)[:,-1:],self.readout_weight)
  return (z[:,0,1]-z[:,0,0]).double().numpy().tolist()
 def margin(self,record,y):return self.margin_many(record,[y])[0]
 def close(self):
  self.subject.close()
  for h in self.handles:h.remove()

def unpack(values):return {k:decode(v).numpy() if isinstance(v,dict) else np.asarray(v) for k,v in values.items()}
def state(record):return unpack(record['state'])
def scalar(s):return float(s['logits'][1])-float(s['logits'][0])
