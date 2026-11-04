import json
from research_checks.gate import sha
from analysis.A_transfer_next_20261103 import worker
O=worker.BASE/'run';before={p.relative_to(O).as_posix():sha(p) for p in O.rglob('*') if p.is_file()}
def forbidden(*a,**kw):raise AssertionError('Completed replay attempted native model load')
old=worker.load;worker.load=forbidden
try:worker.main()
finally:worker.load=old
after={p.relative_to(O).as_posix():sha(p) for p in O.rglob('*') if p.is_file()};assert before==after
v=dict(passed=True,files_unchanged=len(before),new_calls=0,new_model_loads=0,scope='Independent actual worker.main with blocked loader; hashes before/after allrunfiles')
(worker.BASE/'review/independent_replay_audit.json').write_text(json.dumps(v,indent=2)+'\n');print(v)
