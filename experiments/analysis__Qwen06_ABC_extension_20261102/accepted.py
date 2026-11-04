"""Strict accepted-snapshot proof check; builders bind every consumed byte.

Avoid re-reading unused full checkpoints during metadata assembly. The actual
launcher independently verifies every new packet input before importing worker.
"""
from research_checks.gate import digest
from analysis.C2_fresh_fit_holdout_20261102.worker import ROOT,read
def accepted(snapshot,review):
 s=read(ROOT/snapshot);r=read(ROOT/review);p=s['payload'];assert p['phase']=='completion' and digest(p)==s['snapshot_sha256']
 assert set(r)=={'schema_version','snapshot_sha256','reviewer_id','independent','verdict','topics_checked','findings','limitations'} and r['schema_version']==1 and r['snapshot_sha256']==s['snapshot_sha256'] and r['independent'] is True and r['reviewer_id']!=p['primary_agent'] and r['verdict']=='approved' and set(r['topics_checked'])=={'design','implementation','data','cost','claims'} and isinstance(r['limitations'],str) and r['limitations']
 assert all(f['status']!='open' and (f['severity'] not in ['high','blocker'] or f['status']=='resolved') for f in r['findings']);return s
