"""Exact common-prefix reuse for Gemma post-MLP output interventions.

All later native layers execute with dynamic attention; no frozen attention,
KV cache, reduced circuit, or last-layer approximation. B-only new adapter.
"""
import base64
import numpy as np
import torch
from .native import Native

def encode(value):
    if isinstance(value,torch.Tensor):
        a=value.detach().cpu().contiguous().numpy()
        return dict(_tensor=True,dtype=a.dtype.str,shape=list(a.shape),data=base64.b64encode(a.tobytes()).decode('ascii'))
    if isinstance(value,tuple):return dict(_tuple=[encode(x) for x in value])
    if isinstance(value,list):return [encode(x) for x in value]
    if isinstance(value,dict):return {k:encode(v) for k,v in value.items()}
    if value is None or isinstance(value,(bool,int,float,str)):return value
    raise TypeError('Unsupported native tail argument '+type(value).__name__)

def decode(value):
    if isinstance(value,dict) and value.get('_tensor') is True:
        a=np.frombuffer(base64.b64decode(value['data'],validate=True),dtype=np.dtype(value['dtype'])).copy().reshape(value['shape'])
        return torch.from_numpy(a)
    if isinstance(value,dict) and '_tuple' in value:return tuple(decode(x) for x in value['_tuple'])
    if isinstance(value,dict):return {k:decode(v) for k,v in value.items()}
    if isinstance(value,list):return [decode(x) for x in value]
    return value

class ExactTail:
    def __init__(self,model,layer,answer_ids):
        if model.training:raise ValueError('Exact tail requires eval mode')
        self.native=Native(model,layer,answer_ids);self.model=model;self.layer=self.native.layer
        if not 0<=self.layer<len(model.model.layers)-1:raise ValueError('Tail requires actual later layers')
        self._collect=False;self.boundary=None;self.kwargs={};self.handles=[]
        def boundary(module,args,y):
            if self._collect:
                if not isinstance(y,torch.Tensor):raise TypeError('Unexpected decoder output')
                self.boundary=encode(y)
        self.handles.append(model.model.layers[self.layer].register_forward_hook(boundary))
        for idx in range(self.layer+1,len(model.model.layers)):
            def arguments(module,args,kwargs,idx=idx):
                if self._collect:
                    if kwargs.get('past_key_values') is not None:raise ValueError('KV cache forbidden')
                    self.kwargs[str(idx)]=encode(kwargs)
            self.handles.append(model.model.layers[idx].register_forward_pre_hook(arguments,with_kwargs=True))

    def capture(self,ids):
        self._collect=True;self.boundary=None;self.kwargs={}
        try:r=self.native.forward(ids,full_head=True)
        finally:self._collect=False
        expected={str(i) for i in range(self.layer+1,len(self.model.model.layers))}
        if self.boundary is None or set(self.kwargs)!=expected:raise ValueError('Incomplete tail capture')
        return dict(layer=self.layer,answer_ids=list(self.native.answers),ids=list(ids),boundary=self.boundary,kwargs=self.kwargs,
                    states={k:v[0].tolist() for k,v in r.items() if k in ('R','x','y','margin')})

    def tensor_margin(self,record,output_value):
        if record['layer']!=self.layer or record['answer_ids']!=list(self.native.answers):raise ValueError('Incompatible tail record')
        # decode creates ordinary writable tensors, not inference tensors.
        h=decode(record['boundary']);r=torch.tensor(record['states']['R'],dtype=h.dtype)
        value=torch.as_tensor(output_value,dtype=h.dtype)
        if value.ndim!=1 or value.shape!=r.shape or not torch.isfinite(value).all():raise ValueError('Invalid output replacement')
        h=h.clone();h[0,-1]=r+value  # exact native addition order, not (R+M0)+D
        for idx in range(self.layer+1,len(self.model.model.layers)):
            kwargs=decode(record['kwargs'][str(idx)])
            if kwargs.get('past_key_values') is not None:raise ValueError('KV cache forbidden')
            h=self.model.model.layers[idx](h,**kwargs)
        final=self.model.model.norm(h)[:,-1]
        return self.native._margin(final)[0]

    def margin(self,record,output_value):
        with torch.inference_mode():return float(self.tensor_margin(record,output_value))

    def gradient(self,record,output_value):
        with torch.enable_grad():
            leaf=torch.tensor(output_value,dtype=torch.float32,requires_grad=True)
            margin=self.tensor_margin(record,leaf)
            gradient=torch.autograd.grad(margin,leaf)[0]
        return dict(margin=float(margin.detach()),gradient=gradient.detach().double().numpy().tolist())

    def close(self):
        self.native.close()
        for handle in self.handles:handle.remove()

