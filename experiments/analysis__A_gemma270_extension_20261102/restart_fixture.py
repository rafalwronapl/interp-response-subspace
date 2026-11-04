"""Cache-only prefix replay and consumed-attempt failure; no model."""
import json,tempfile
from pathlib import Path
from ab_confirmatory_20261101.runtime import Runtime
from analysis.A_gemma270_extension_20261102.capture import restore_cached_prefix,sha

def main():
 counter=[0];cfg=dict(answer_ids=[1,2]);docs=[dict(id='one'),dict(id='two')];tokens={'one':dict(ids=[1,2]),'two':dict(ids=[3,4])};schedule=dict(operations=[dict(id='capture_'+d['id'],kind='forward') for d in docs]);identity=dict(model='synthetic',code='synthetic')
 with tempfile.TemporaryDirectory() as t:
  root=Path(t);rt=Runtime(root,identity,schedule)
  def once():counter[0]+=1;return dict(R=[1],x=[2],y=[3],margin=4)
  rt.compute('capture_one',dict(document=docs[0],tokens=tokens['one']),dict(mode='natural'),dict(answer_ids=[1,2],orientation='is-minus-are'),once)
  prefix={str(p.relative_to(root)):sha(p) for d in ['cache','completed'] for p in (root/d).glob('*.json')};restore_cached_prefix(Runtime(root,identity,schedule),docs,tokens,cfg);assert counter[0]==1 and all(sha(root/n)==h for n,h in prefix.items())
  def interruption():raise RuntimeError('synthetic interruption after charged reservation')
  try:rt.compute('capture_two',dict(document=docs[1],tokens=tokens['two']),dict(mode='natural'),dict(answer_ids=[1,2],orientation='is-minus-are'),interruption)
  except RuntimeError:pass
  else:raise AssertionError('Expected interrupted reservation')
  try:restore_cached_prefix(Runtime(root,identity,schedule),docs,tokens,cfg)
  except ValueError as e:assert 'consumed operation' in str(e)
  else:raise AssertionError('Unknown consumed operation did not fail closed')
  assert counter[0]==1
 p=Path(__file__).parent;(p/'restart_fixture_v1.json').write_text(json.dumps(dict(status='PASS',model_loads=0,native_forward=0,checks=['Durable prefix replay preserves bytes with0newcallbacks','Unknown charged attempt without cache fails before model loader'],sources={n:sha(p/n) for n in ['capture.py','restart_fixture.py']}),indent=2)+'\n');print('Restart fixtures PASS,0modelcalls')
if __name__=='__main__':main()
