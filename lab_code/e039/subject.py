"""Native upstream/downstream with an optional target-free MLP21 bypass."""
import numpy as np
import torch
from e036.subject import Subject as NativeSubject, Atlas


class Subject(NativeSubject):
    def __init__(self,model,xindex=1317,mindex=1423):
        super().__init__(model,xindex,mindex)
        self.projection_calls=np.zeros(3,dtype=np.int64)
        mlp=model.model.layers[21].mlp
        for i,name in enumerate(('gate_proj','up_proj','down_proj')):
            def count(module,args,i=i):
                self.projection_calls[i]+=1
            self.handles.append(getattr(mlp,name).register_forward_pre_hook(count))
        def capture_output(module,args,out):
            if self.state is not None:self.state['Y']=out[0].detach().numpy().copy()
        self.handles.append(mlp.register_forward_hook(capture_output))

    def forward(self,doc,prefix=False,x=None,m=None,replacement=None,poison=False,negative=False):
        target=self.model.model.layers[21].mlp;original=target.forward
        self.projection_calls[:]=0
        def bypass(h,*args,**kwargs):
            if replacement is not None:
                y=torch.as_tensor(replacement,dtype=h.dtype,device=h.device)
                if y.shape!=h[0].shape:raise ValueError('Replacement sequence/hidden shape')
                return y.unsqueeze(0).clone()
            if poison:raise AssertionError('Forbidden original MLP21 forward')
            return original(h,*args,**kwargs)
        target.forward=bypass
        try:
            result=super().forward(doc,prefix=prefix,x=x,m=m,negative=negative)
            result['projection_calls']=self.projection_calls.copy()
            expected=0 if prefix or replacement is not None else 1
            if not np.array_equal(self.projection_calls,np.full(3,expected)):
                raise AssertionError('MLP21 component execution mismatch')
            return result
        finally:target.forward=original

    def readout_direction(self):
        w=self.model.lm_head.weight
        is_id,are_id=(374,525) if w.shape[0]>525 else (0,1)
        direction=(w[are_id]-w[is_id]).detach().float().numpy().copy()
        norm=float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm<=0:raise ValueError('Undefined readout direction')
        return direction/norm
