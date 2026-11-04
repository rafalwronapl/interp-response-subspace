"""Original Qwen hooks; prefix runs stop BEFORE calling the MLP21 module."""
import numpy as np
import torch
from .common import ROOT, array_id

class BeforeTarget(Exception): pass

class Subject:
    def __init__(self,model,xindex=1317,mindex=1423):
        self.model=model;self.state=None;self.handles=[];self.xindex=xindex;self.mindex=mindex
        for layer,index,name in [(19,xindex,'X'),(20,mindex,'M')]:
            def pre(module,args,layer=layer,index=index,name=name):
                if self.state is None:return
                z=args[0];self.state[name]=z[0,-1,index].detach().numpy().copy()
                target=self.settings[name]
                if target is None or self.settings['style']=='post':return
                patched=z.clone();patched[0,-1,index]=target
                if not torch.equal(patched[:,:-1],z[:,:-1]):raise RuntimeError('Patch position error')
                return (patched,)+args[1:]
            self.handles.append(model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(pre))
            def post(module,args,out,layer=layer,index=index,name=name):
                if self.state is None:return
                target=self.settings[name]
                if target is None or self.settings['style']!='post':return
                out=out.clone();out[0,-1]+=module.down_proj.weight[:,index]*(target-float(self.state[name]))
                return out
            self.handles.append(model.model.layers[layer].mlp.register_forward_hook(post))
        for layer in (19,20,21):
            def capture(module,args,layer=layer):
                if self.state is None:return
                h=args[0];self.state['x'+str(layer)]=h[0,-1].detach().numpy().copy()
                self.state['earlier'+str(layer)]=np.asarray(array_id({'h':h[:,:-1].detach().numpy()})['h'])
                if layer==21 and self.settings['prefix']:raise BeforeTarget()
            self.handles.append(model.model.layers[layer].mlp.register_forward_pre_hook(capture))
        def after(module,args,out):
            if self.state is None:return
            self.state['y']=out[0,-1].detach().numpy().copy()
            if self.settings['negative']:
                out=out.clone();out[0,-1]+=0.1
                return out
        self.handles.append(model.model.layers[21].mlp.register_forward_hook(after))
    def forward(self,doc,prefix=False,x=None,m=None,style='pre',negative=False):
        self.state={};self.settings=dict(X=x,M=m,prefix=prefix,style=style,negative=negative)
        try:
            with torch.inference_mode():
                try:
                    result=self.model(torch.tensor([doc['ids']]),use_cache=False)
                    self.state['logits']=result.logits[0,-1,[374,525]].detach().numpy().copy() if result.logits.shape[-1]>525 else result.logits[0,-1,:2].detach().numpy().copy()
                except BeforeTarget:
                    if not prefix:raise
            if prefix and ('y' in self.state or 'logits' in self.state):raise RuntimeError('Target leakage')
            return self.state.copy()
        finally:self.state=None
    def local(self,x,dtype=torch.float32,tokens=1):
        mlp=self.model.model.layers[21].mlp
        if tokens<1:raise ValueError('Local batch length')
        h=torch.zeros((1,tokens,len(x)),dtype=dtype)
        h[0,-1]=torch.as_tensor(x,dtype=dtype)
        with torch.inference_mode():
            g=torch.nn.functional.linear(h,mlp.gate_proj.weight.to(dtype))
            u=torch.nn.functional.linear(h,mlp.up_proj.weight.to(dtype))
            return torch.nn.functional.linear(torch.nn.functional.silu(g)*u,mlp.down_proj.weight.to(dtype))[0,-1].numpy().copy()
    def jvp(self,x,delta,dtype=torch.float32):
        mlp=self.model.model.layers[21].mlp
        with torch.inference_mode():
            h=torch.as_tensor(x,dtype=dtype);dx=torch.as_tensor(delta,dtype=dtype)
            g=mlp.gate_proj.weight.to(dtype)@h;u=mlp.up_proj.weight.to(dtype)@h
            dg=mlp.gate_proj.weight.to(dtype)@dx;du=mlp.up_proj.weight.to(dtype)@dx
            sig=torch.sigmoid(g)
            return (mlp.down_proj.weight.to(dtype)@(sig*(1+g*(1-sig))*dg*u+torch.nn.functional.silu(g)*du)).numpy().copy()
    def close(self):
        for h in self.handles:h.remove()

class Atlas:
    def __init__(self):
        from safetensors import safe_open
        self.layers={}
        for layer in (20,21):
            with safe_open(str(ROOT/f'clt_cache/e017/W_enc_{layer}.safetensors'),framework='pt',device='cpu') as f:
                weights={k:f.get_tensor(k+'_'+str(layer)).float() for k in ('W_enc','b_enc','threshold')}
            with safe_open(str(ROOT/f'clt_cache/e033/W_dec_{layer}.safetensors'),framework='pt',device='cpu') as f:
                s=f.get_slice('W_dec_'+str(layer))
                if s.get_shape()!=[20480,28-layer,1024]:raise ValueError('Decoder axes')
                weights['decoder']=s[:,21-layer,:].float()
            if list(weights['W_enc'].shape)!=[20480,1024]:raise ValueError('Encoder axes')
            self.layers[layer]=weights
    def evaluate(self,state,reference=None):
        result={};value=None
        with torch.inference_mode():
            for layer,w in self.layers.items():
                p=torch.nn.functional.linear(torch.as_tensor(state['x'+str(layer)]),w['W_enc'],w['b_enc'])
                f=p*(p>w['threshold'])
                df=f if reference is None else f-torch.as_tensor(reference['f'+str(layer)])
                v=w['decoder'].T@df
                value=v if value is None else value+v
                result['f'+str(layer)]=f.numpy().copy()
            result['p']=value.numpy().copy()
        return result
