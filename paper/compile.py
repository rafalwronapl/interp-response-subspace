"""Compile main and supplement without installations or scientific calculations."""
from pathlib import Path
import subprocess,json,sys
from pypdf import PdfReader
here=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
statuses={}
for name in ['main','supplement']:
 codes=[]
 for i,cmd in enumerate([
  ['pdflatex','--disable-installer','-interaction=nonstopmode','-halt-on-error',name+'.tex'],
  ['bibtex',name],
  ['pdflatex','--disable-installer','-interaction=nonstopmode','-halt-on-error',name+'.tex'],
  ['pdflatex','--disable-installer','-interaction=nonstopmode','-halt-on-error',name+'.tex']]):
  p=subprocess.run(cmd,cwd=here,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
  (here/f'{name}_build_{i}.txt').write_text(p.stdout+'\n'+p.stderr,encoding='utf-8');codes.append(p.returncode)
  if p.returncode:raise RuntimeError(p.stdout[-3000:]+p.stderr[-500:])
 doc=PdfReader(here/(name+'.pdf'));text='\n'.join(p.extract_text() or '' for p in doc.pages)
 (here/(name+'_extracted_text.txt')).write_text(text,encoding='utf-8')
 log=(here/(name+'.log')).read_text(errors='replace')
 status=dict(exit_codes=codes,pages=len(doc.pages),undefined_references='There were undefined references' in log,undefined_citations='Citation' in log and 'undefined' in log,overfull_boxes='Overfull' in log)
 statuses[name]=status
(here/'build_status.json').write_text(json.dumps(statuses,indent=2)+'\n',encoding='utf-8');print(json.dumps(statuses))
