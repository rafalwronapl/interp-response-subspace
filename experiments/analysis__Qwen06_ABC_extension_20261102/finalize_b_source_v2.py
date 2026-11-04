"""Pre-measurement B baseline/accounting alignment with actual original B."""
from pathlib import Path
base=Path(__file__).parent
p=base/'prepare_b.py';t=p.read_text(encoding='utf-8').replace('6methods6doses','7methods6doses').replace('forward=92824','forward=96920').replace("BASE/'b_fixture_v1.json'","BASE/'b_fixture_v2.json'").replace("BASE/'scoring_fixture_v1.json',BASE/'scoring_fixtures.py'","BASE/'b_scoring_fixture_v2.json',BASE/'b_scoring_fixtures_v2.py'").replace("'b_fixtures.py','prepare_b.py'","'b_fixtures_v2.py','prepare_b.py'");p.write_text(t,encoding='utf-8')
t=(base/'b_fixtures.py').read_text(encoding='utf-8').replace('branch_jvp,clt_delta','branch_jvp,clt_delta,clt_response').replace("'b_fixtures.py'","'b_fixtures_v2.py'").replace("'b_fixture_v1.json'","'b_fixture_v2.json'")
needle=' headers={}'
extra=" tc,fr=clt_response(x,x+dx,dx,a);oldtc,oldfr=clt_delta(x,x+dx,a);assert torch.equal(tc,oldtc) and torch.allclose(fr,oldfr,atol=1e-12,rtol=0)\n"
assert needle in t;t=t.replace(needle,extra+needle);(base/'b_fixtures_v2.py').write_text(t,encoding='utf-8')
print('BstrongTaylor/frozen-direction alignment finalized; oldQfixtures preserved')
