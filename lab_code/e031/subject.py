"""Differentiable native U/W interventions; restore MLP20 after down_proj."""
import time
import numpy as np
import torch
from e024.common import sha_array

UNITS={'U':(19,4811),'W':(21,972)}


class Subject:
    def __init__(self,budget,model):
        self.budget=budget;self.model=model.eval().requires_grad_(False)
        if any(p.dtype!=torch.float32 or p.device.type!='cpu' for p in self.model.parameters()):raise ValueError('CPU FP32 required')
        self.handles=[];self.settings={};self.capture={};self.audits=[]
        for name,(layer,index) in UNITS.items():
            def unit(mod,args,name=name,index=index):
                z=args[0];self.capture[name]=np.asarray(float(z[0,-1,index].detach()),dtype=np.float32)
                value=self.settings.get(name)
                if value is None:return
                before=z;z=z.clone();z[0,-1,index]=value
                with torch.no_grad():
                    mask=torch.zeros_like(z,dtype=torch.bool);mask[0,-1,index]=True
                    if not torch.equal(z[~mask],before[~mask]):raise RuntimeError('Wrong unit coordinates')
                    expected=torch.as_tensor(value,dtype=z.dtype)
                    if not torch.equal(z[0,-1,index],expected):raise RuntimeError('Unit replacement mismatch')
                self.audits.append(dict(kind='unit',unit=name,layer=UNITS[name][0],index=index,position=self.last,
                    replacement=float(z[0,-1,index].detach()),outside_exact=True))
                return (z,)+args[1:]
            self.handles.append(self.model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(unit))
            for part in ('gate','up'):
                def capture(mod,args,out,name=name,index=index,part=part):self.capture[name+'_'+part]=np.asarray(float(out[0,-1,index].detach()),dtype=np.float32)
                self.handles.append(getattr(self.model.model.layers[layer].mlp,part+'_proj').register_forward_hook(capture))
        def mlp20(mod,args,out):
            self.capture['MLP20_before']=out[0,-1].detach().numpy().copy();before=out
            value=self.settings.get('MLP20')
            if value is not None:
                target=torch.as_tensor(value,dtype=out.dtype)
                if target.shape!=(out.shape[-1],):raise ValueError('MLP20 target shape')
                out=out.clone();out[0,-1]=target
                if not torch.equal(out[:,:-1],before[:,:-1]) or not torch.equal(out[0,-1],target):raise RuntimeError('MLP20 replacement mismatch')
                self.audits.append(dict(kind='MLP20',position=self.last,outside_exact=True,target_sha256=sha_array(np.asarray(value))))
            self.capture['MLP20_after']=out[0,-1].detach().numpy().copy();return out
        self.handles.append(self.model.model.layers[20].mlp.down_proj.register_forward_hook(mlp20))
        def norm(mod,args,out):
            self.capture['final_hidden']=out[0,-1].detach().numpy().copy()
            self.capture['prefix_hash']=np.asarray(sha_array(out[:,:-1].detach().numpy()))
        self.handles.append(self.model.model.norm.register_forward_hook(norm))

    def run(self,doc,label,reference=None,t=None,delta=None,branch='b',derivatives=False,U_override=None,W_override=None,clamp_mlp20=True):
        if branch not in ('b','c') or (derivatives and t!=0.):raise ValueError('Invalid branch/derivative origin')
        if reference is None and any(x is not None for x in (t,delta,U_override,W_override)):raise ValueError('Missing natural reference')
        self.capture={};self.audits=[];self.settings={};self.last=len(doc['ids'])-1
        scalar=None
        if reference is not None:
            if clamp_mlp20:self.settings['MLP20']=reference['MLP20_after']
            if t is not None:
                if delta is None:raise ValueError('Missing natural donor delta')
                scalar=torch.tensor(float(t),dtype=torch.float32,requires_grad=derivatives)
                self.settings['U']=torch.tensor(float(reference['U']),dtype=torch.float32)+scalar*float(delta)
            if branch=='c':self.settings['W']=float(reference['W'])
            if U_override is not None:self.settings['U']=float(U_override)
            if W_override is not None:self.settings['W']=float(W_override)
        ticket=self.budget.reserve('forward',label);start=time.monotonic();ok=False
        try:
            with torch.set_grad_enabled(derivatives):
                logits=self.model(input_ids=torch.tensor([doc['ids']]),attention_mask=torch.ones((1,len(doc['ids'])),dtype=torch.long),use_cache=False,logits_to_keep=1).logits[0,-1]
                readout=doc.get('readout_ids',[374,525]);y=logits[readout[0]]-logits[readout[1]]
                if y.dtype!=torch.float32 or not torch.isfinite(logits).all():raise ValueError('Invalid forward precision/output')
            rec=dict(margin=float(y.detach()),scores=[float(logits[i].detach()) for i in readout],audits=self.audits.copy(),
                     branch=branch,t=t,delta=delta,derivatives=derivatives,readout=readout)
            arrays={k:v.copy() for k,v in self.capture.items()};ok=True
        finally:self.budget.finish(ticket,ok,time.monotonic()-start)
        try:
            if derivatives:
                if scalar is None or not y.requires_grad:raise RuntimeError('Detached active graph')
                values={};gradient=None
                for name in ('prime','second'):
                    backward_label=label.removesuffix('_zero')+'_'+name
                    ticket=self.budget.reserve('backward',backward_label);start=time.monotonic();ok=False
                    try:
                        gradient=torch.autograd.grad(y if name=='prime' else gradient,scalar,create_graph=name=='prime')[0]
                        if not torch.isfinite(gradient):raise ValueError('Nonfinite derivative')
                        values[name]=float(gradient.detach());ok=True
                    finally:self.budget.finish(ticket,ok,time.monotonic()-start)
                rec.update(values)
            if reference is not None:
                if str(arrays['prefix_hash'])!=str(reference['prefix_hash']):raise RuntimeError('Earlier final positions changed')
                if clamp_mlp20:np.testing.assert_array_equal(arrays['MLP20_after'],reference['MLP20_after'])
            return rec,arrays
        finally:self.capture={};self.settings={};self.audits=[]

    def close(self):
        for h in self.handles:h.remove()
        self.handles=[]
