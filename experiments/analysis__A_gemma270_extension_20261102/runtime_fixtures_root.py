"""Admission invariants only; no launcher, model or real lock mutation."""
import json,os,tempfile
from pathlib import Path
import concurrency_a270_q06_20261102 as m

def main():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory); lane=root/'operations/concurrency_runtime_v1';lane.mkdir(parents=True)
        def put(p,x):p.write_text(json.dumps(x))
        put(lane/'model.lock',dict(pid=os.getpid(),snapshot_sha256='gemma'))
        qlock=root/'operations/research_worker.lock'
        put(qlock,dict(token='qtoken',snapshot_sha256='qsha'))
        procs=[dict(ProcessId=10,ParentProcessId=1,CreationDate='a',CommandLine='-m research_checks launch q'),
               dict(ProcessId=11,ParentProcessId=10,CreationDate='b',CommandLine='-m research_checks._worker q')]
        policy=dict(legacy_cpu=dict(processes=procs,lock_sha256=m.sha(qlock),token='qtoken',snapshot_sha256='qsha',snapshot='s',review='r'),
                    parallel_native=dict(stage_snapshot_sha256='gemma',legacy_budget=dict(forward=2844,backward=0),legacy_entrypoint='q'))
        originalverify=m.verify_review;originalcontract=m.contract
        m.verify_review=lambda *a:dict(snapshot_sha256='qsha',payload=dict(contract='c'))
        m.contract=lambda *a:dict(budget=dict(forward=2844,backward=0),entrypoint=dict(module='q'))
        table={p['ProcessId']:dict(p) for p in procs}
        assert m.legacy_state(root,policy,table)=='exact_reviewed_parallel_native'
        checks=['exact approved Q plus exact Gemma admitted']
        def reject(label,table):
            try:m.legacy_state(root,policy,table)
            except ValueError:checks.append(label)
            else:raise AssertionError(label)
        bad=dict(table);bad[12]=dict(ProcessId=12,CommandLine='-m research_checks launch extra');reject('third launcher rejected',bad)
        reject('missing Q child rejected',{10:table[10]})
        put(lane/'model.lock',dict(pid=os.getpid(),snapshot_sha256='other'));reject('unreviewed Gemma stage rejected',table)
        put(lane/'model.lock',dict(pid=os.getpid(),snapshot_sha256='gemma'))
        put(lane/'cpu.lock',{});reject('extra CPU lane rejected',table);(lane/'cpu.lock').unlink()
        qlock.unlink();reject('live Q without global lock rejected',table)
        assert m.legacy_state(root,policy,{}) is None;checks.append('finished Q permits canonical global lock acquisition')
        m.verify_review=originalverify;m.contract=originalcontract
    out=Path(__file__).with_name('runtime_fixtures_root_v1.json')
    out.write_text(json.dumps(dict(status='pass',checks=checks),indent=2)+'\n')
    print(json.dumps(dict(status='pass',checks=checks)))

if __name__=='__main__':main()
