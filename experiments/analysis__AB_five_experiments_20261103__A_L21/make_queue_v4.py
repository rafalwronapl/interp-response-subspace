from pathlib import Path
p=Path(__file__).parent
s=(p/'queue_pilot.py').read_text().replace('v3','v4').replace('queue_status','queue_status_v4').replace("pilot_run/completed.json","pilot_run_v4/completed.json")
(p/'queue_pilot_v4.py').write_text(s)
