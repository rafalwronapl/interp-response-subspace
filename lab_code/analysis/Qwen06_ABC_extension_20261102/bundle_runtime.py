"""Stage-local coupled-operation accounting with one immutable batch receipt.

All logical forward IDs remain in the frozen schedule and durable attempt.
Only the number of filesystem nodes changes; measurement callbacks are literal.
"""
import time
from ab_confirmatory_20261101.runtime import Runtime as BaseRuntime
from research_checks.persistence import cached,immutable,request_key

class BundleRuntime(BaseRuntime):
 def __init__(self,folder,identity,schedule):
  super().__init__(folder,identity,schedule)
  self.attempts=set();self.batches={}
  for record in self.records:
   if record['event']!='attempt':continue
   ops=record['coupled_operations'];assert ops and len(set(ops))==len(ops) and all(op in self.ops and self.ops[op]=='forward' for op in ops)
   assert not self.attempts.intersection(ops),'Overlapping durable attempt records'
   assert record['operation']==ops[0] and record['logical_forward']==len(ops)
   self.attempts.update(ops);self.batches[record['request']]=tuple(ops)

 def compute(self,operation,document,intervention,readout,function,extra_operations=()):
  operations=[operation,*extra_operations]
  if len(set(operations))!=len(operations) or not operations:raise ValueError('Duplicate/empty coupled operation')
  if any(op not in self.ops or self.ops[op]!='forward' for op in operations):raise ValueError('Unknown/non-forward coupled operation')
  if 'logical_operations' in intervention:raise ValueError('Reserved bundle identity field')
  request=dict(model_identity=self.identity['model'],code_identity=self.identity['code'],document=document,intervention=dict(intervention,logical_operations=operations),readout=readout);key=request_key(request)
  if key in self.batches and self.batches[key]!=tuple(operations):raise ValueError('Changed coupled batch')
  def measured():
   if any(op in self.attempts for op in operations):raise ValueError('Consumed operation without durable cache; reviewed retry needed')
   self.record(dict(event='attempt',operation=operation,coupled_operations=operations,kind='forward',logical_forward=len(operations),request=key))
   self.attempts.update(operations);self.batches[key]=tuple(operations);started=time.perf_counter();result=function()
   self.record(dict(event='computed',operation=operation,coupled_operations=operations,logical_forward=len(operations),request=key,seconds=time.perf_counter()-started));return result
  result=cached(self.folder/'cache',request,measured)
  if key not in self.batches or self.batches[key]!=tuple(operations):raise ValueError('Unaccounted durable cache')
  immutable(self.folder/'bundles'/(operation+'.json'),dict(operations=operations,logical_forward=len(operations),request=key))
  return result
