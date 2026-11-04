"""Random-weight qualification of independent rows in exact Qwen suffix batches."""
from pathlib import Path
import json,numpy as np,torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from .adapter_v4 import Adapter,state,scalar,unpack
from research_checks.gate import sha
BASE=Path(__file__).parent
def main():
 torch.set_num_threads(2);torch.manual_seed(31103)
 cfg=Qwen3Config(vocab_size=600,hidden_size=16,intermediate_size=1440,num_hidden_layers=28,num_attention_heads=2,num_key_value_heads=2,head_dim=8);cfg._attn_implementation='eager'
 model=Qwen3ForCausalLM(cfg).eval();a=Adapter(model);d=dict(id='synthetic',ids=[1,4,7,9]);base=a.full(d);s=state(base);x=float(s['X'])+.3;m=float(s['M']);native=a.full(d,x,m);t=state(native)
 p=unpack(a.prefix(d,x,m));assert all(np.array_equal(p[k],t[k]) for k in ['x19','x20','x21'])
 targets=[t['y'],s['y']]+[np.asarray(t['y']+k*.01*np.arange(16),np.float32) for k in range(1,15)]
 full=[]
 for y in targets:
  Y=s['Y'].copy();Y[-1]=y;full.append(scalar(state(a.full(d,x,m,Y))))
 result=a.margin_many(native,targets);errors=np.abs(np.asarray(full)-result)
 assert max(errors)<1e-5
 singles=[a.margin(native,y) for y in targets];assert np.max(np.abs(np.asarray(singles)-result))<1e-5
 assert max(result)-min(result)>1e-6
 a.close();data=dict(ok=True,random_weights_only=True,trained_calls=0,targets=16,full_batch_maxabs=float(max(errors)),single_batch_maxabs=float(np.max(np.abs(np.asarray(singles)-result))),adapter_sha256=sha(BASE/'adapter_v4.py'))
 (BASE/'batch_fixture.json').write_text(json.dumps(data,indent=2));print(data)
if __name__=='__main__':main()
