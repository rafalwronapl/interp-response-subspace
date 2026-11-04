from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent
h=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
old=p.parent/'A_v12_20261103'/'evidence'
pres=[]
for f in old.rglob('*'):
 if f.is_file():
  dest=p/'evidence'/f.relative_to(old)
  assert dest.read_bytes()==f.read_bytes(),str(f)
  pres.append({'path':str(f.relative_to(old)).replace('\\','/'),'sha256':h(f)})
(p/'evidence_preservation.json').write_text(json.dumps({'prior':'A_v12_20261103/evidence','byte_exact':True,'files':pres},indent=2)+'\n')
(p/'EVIDENCE_MANIFEST.json').write_text(json.dumps({'files':{str(f.relative_to(p)).replace('\\','/'):h(f) for f in sorted((p/'evidence').rglob('*')) if f.is_file()},'v13_source_manifest':'v13_source_manifest.json','large_raw_caches_forecasts_models':'omitted; canonical experiment manifests bind original sources'},indent=2)+'\n')
(p/'README.txt').write_text('A13 private working manuscript, November 3, 2026. Compile with python compile.py. New evidence in evidence/v13; complete source/copy hashes in v13_source_manifest.json. Prior A12 evidence preserved byte-exact. prior_v12_review applies only to the prior version. Fresh independent A13 editorial review pending.\n')
(p/'editorial_checks.json').write_text(json.dumps({'version':13,'pages':19,'compile_exit_codes':[0,0,0,0],'undefined_references':False,'undefined_citations':False,'overfull_boxes':False,'actual_pdf_read':'Extracted all pages; visually inspected pages 1,7,9,10 including new tables and semantic correction.','prior_evidence_byte_exact':True,'new_scientific_computation':False,'independent_editorial_review':'pending'},indent=2)+'\n')
files={str(f.relative_to(p)).replace('\\','/'):h(f) for f in sorted(p.rglob('*')) if f.is_file() and 'review' not in f.relative_to(p).parts and f.name!='editorial_snapshot.json' and '__pycache__' not in f.parts}
snap={'revision':'A_v13_20261103','scope':'New manuscript version integrating three completed approved campaigns; no new scientific computation.','independent_review_status':'pending','files':files}
(p/'editorial_snapshot.json').write_text(json.dumps(snap,indent=2)+'\n')
print(json.dumps({'snapshot_sha256':h(p/'editorial_snapshot.json'),'pdf_sha256':h(p/'main.pdf'),'files':len(files),'prior_evidence_files':len(pres)}))
