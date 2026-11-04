"""Replay exactly materialized coefficients without solving again; reject drift."""
from pathlib import Path
from unittest.mock import patch
from . import models as m
from research_checks.persistence import immutable
def main():
    before={n:m.sha(m.BASE/n) for n in ['MODELS.json','model_checks.json']}
    with patch.object(m,'fit',side_effect=AssertionError('Duplicate solve')):m.main()
    assert all(m.sha(m.BASE/n)==h for n,h in before.items())
    actual=m.sha
    def changed(p):return 'incompatible-code' if Path(p).resolve()==Path(m.__file__).resolve() else actual(p)
    with patch.object(m,'sha',changed),patch.object(m,'fit',side_effect=AssertionError('Solve before identity check')):
        try:m.main()
        except ValueError as e:assert 'identity changed' in str(e)
        else:raise AssertionError('Incompatible materialization accepted')
    immutable(m.BASE/'model_replay_checks.json',dict(passed=True,replay_new_solves=0,preserved_hashes=before,incompatible_identity_rejected_before_fit=True))
    print('model coefficient replay / identity PASS')
if __name__=='__main__':main()
