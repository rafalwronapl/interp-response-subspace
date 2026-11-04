"""Use reproducible FP64 vector hashes instead of redundant dense disk files."""
from pathlib import Path
BASE=Path(__file__).parent
p=BASE/'evaluate.py';t=p.read_text(encoding='utf-8')
helper='''
def vector_hash(values):
 h=hashlib.sha256()
 for k in sorted(values):
  a=np.ascontiguousarray(values[k]);h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()

'''
t=t.replace('def run(config):',helper+'def run(config):')
t=t.replace("  publish_npz(file,values)\n  noD_contexts.append(dict(document=d['id'],path=file.relative_to(ROOT).as_posix(),sha256=sha(file)))", "  noD_contexts.append(dict(document=d['id'],vector_sha256=vector_hash(values)))")
insert='''
 def reproduce_noD(d,context):
  r=get(d['id']);updates,gradients,_=noD_predict(extract_noD_features(r,get(d['donor'])),ps,residual,pooled,us,assets)
  v=dict(R=np.asarray(r['R'],np.float32),y=np.asarray(r['y'],np.float32),x_rec=np.asarray(r['x']),x_donor=np.asarray(get(d['donor'])['x']),**updates,**gradients)
  assert vector_hash(v)==context['vector_sha256'];return v
 def reproduce_oracle(d,n,context):
  D=np.asarray(get(d['donor'])['y'])-get(d['id'])['y'];v={'native':D}
  for i in FITS:
   p=n['P_'+i];res=D-p
   for j in FITS:
    c=(res@us[j])@us[j].T;v['oracle_'+i+'_'+j]=p+c;v['outside_'+i+'_'+j]=D-c
  assert vector_hash(v)==context['vector_sha256'];return v
'''
t=t.replace(" oracle_records=[];oracle_contexts=[];oracle_dir=",insert+"\n oracle_records=[];oracle_contexts=[];oracle_dir=")
t=t.replace("  with np.load(ROOT/context['path'],allow_pickle=False) as z:original={k:z[k] for k in z.files}","  original=reproduce_noD(d,context)")
t=t.replace("  publish_npz(file,values)\n  oracle_contexts.append(dict(document=d['id'],path=file.relative_to(ROOT).as_posix(),sha256=sha(file)))", "  oracle_contexts.append(dict(document=d['id'],vector_sha256=vector_hash(values)))")
t=t.replace("  with np.load(ROOT/ncontext['path'],allow_pickle=False) as f:n={k:f[k] for k in f.files}\n  with np.load(ROOT/ocontext['path'],allow_pickle=False) as f:o={k:f[k] for k in f.files}", "  n=reproduce_noD(d,ncontext);o=reproduce_oracle(d,n,ocontext)")
t=t.replace("context_sha256=ncontext['sha256'],oracle_context_sha256=ocontext['sha256']", "context_sha256=ncontext['vector_sha256'],oracle_context_sha256=ocontext['vector_sha256']")
assert "context['path']" not in t
p.write_text(t,encoding='utf-8');print('Compact sealed vector hashes; FP64 exact regeneration asserted before finite heads')
