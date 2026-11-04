"""Random weights only; no pretrained model or scientific corpus."""
from pathlib import Path
import json,tempfile,numpy as np,torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from .adapter import Adapter,state,scalar,unpack
from .math import corrections
from ab_confirmatory_20261101.runtime import Runtime

def main():
 torch.set_num_threads(2);torch.manual_seed(31103)
 cfg=Qwen3Config(vocab_size=600,hidden_size=16,intermediate_size=1440,num_hidden_layers=28,num_attention_heads=2,num_key_value_heads=2,head_dim=8)
 cfg._attn_implementation='eager';model=Qwen3ForCausalLM(cfg).eval();a=Adapter(model)
 d=dict(id='synthetic',ids=[1,4,7,9]);base=a.full(d);s=state(base)
 x=float(s['X'])+.3;m=float(s['M']);p=unpack(a.prefix(d,x,m));native=a.full(d,x,m);t=state(native)
 assert all(np.array_equal(p[k],t[k]) for k in ['x19','x20','x21','earlier19','earlier20','earlier21'])
 errors=[]
 for y in [s['y'],t['y'],t['y']+.1*np.arange(16)]:
  Y=s['Y'].copy();Y[-1]=y
  full=state(a.full(d,x,m,Y));tail=a.margin(native,y)
  errors.append(abs(tail-scalar(full)))
 assert max(errors)<1e-5
 assert abs(a.margin(native,t['y']+.1*np.arange(16))-scalar(t))>1e-6
 assert np.array_equal(s['Y'][:-1],t['Y'][:-1]);a.close()
 rng=np.random.default_rng(4);U=np.linalg.qr(rng.normal(size=(40,8)))[0];V=np.linalg.qr(rng.normal(size=(40,8)))[0];e=rng.normal(size=40)
 ca,cb,cc=corrections(e,U,V);assert abs(np.linalg.norm(ca)-np.linalg.norm(cc))<1e-10
 assert corrections(np.zeros(40),U,V)[2] is None
 with tempfile.TemporaryDirectory() as tmp:
  schedule=dict(operations=[dict(id='a',kind='forward'),dict(id='b',kind='forward')]);ident=dict(model='fixture',code='fixture')
  rt=Runtime(tmp,ident,schedule);value=rt.compute('a',d,dict(role='a'),dict(ids=[374,525]),lambda:3.)
  rt2=Runtime(tmp,ident,schedule);assert rt2.compute('a',d,dict(role='a'),dict(ids=[374,525]),lambda:(_ for _ in ()).throw(Exception('replay')))==value
  try:rt2.compute('b',d,dict(role='b'),dict(ids=[374,525]),lambda:(_ for _ in ()).throw(RuntimeError('crash')))
  except RuntimeError:pass
  try:Runtime(tmp,ident,schedule).compute('b',d,dict(role='b'),dict(ids=[374,525]),lambda:0.)
  except ValueError:pass
  else:raise AssertionError('Unreviewed repeat')
 out=dict(hooks=True,data=True,pilot=True,accounting=True,restart=True,random_only=True,tail_full_maxabs=max(errors),trained_calls=0)
 (Path(__file__).parent/'fixtures.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
