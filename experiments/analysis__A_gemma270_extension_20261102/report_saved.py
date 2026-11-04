"""Copy accepted saved scalar output into a compact report; never resamples."""
from pathlib import Path
import json
from analysis.C2_fresh_fit_holdout_20261102.worker import receipt,sha,put
BASE=Path(__file__).parent

def main():
 out=BASE/'evaluation_v1/run';result=receipt(out/'analysis_result.json');summary=dict(frames={k:{n:v[n] for n in ['blocks','dependent_records','losses','contrasts','per_fit']} for k,v in result['frames'].items()},source_sha256={'analysis_result':sha(out/'analysis_result.json'),'head_result':sha(out/'result.json')},scope=result['scope'])
 lines=['# Gemma270: warunkowosc naprawy podprzestrzennej','', '256 nowych blokow leksykalnych, T2/T5, cztery warunki/rame i trzy zachowane pary P/U0. Bez nowego FIT. Jednostka analizy: blok; trzy FIT i cztery warunki usredniono wewnatrz bloku. Primary: T5 inside-P. T2 oraz inside-outside pozostaja opisowe.','', '| Rama | P MSE | inside MSE | outside MSE | inside-P | nominal95% CI |','|---|---:|---:|---:|---:|---|']
 for frame in ['T5','T2']:
  v=result['frames'][frame];s=v['contrasts']['inside_minus_P'];loss=v['losses'];lines.append('| '+frame+' | '+ ' | '.join(f'{loss[a]:.9g}' for a in ['P','inside','outside'])+f" | {s.get('estimate','undefined')} | {s.get('ci',s.get('reason','undefined'))} |")
 lines+=['','## Inside-outside: kontrast opisowy','', '| Rama | inside-outside | nominal95% CI |','|---|---:|---|']
 for frame in ['T5','T2']:
  s=result['frames'][frame]['contrasts']['inside_minus_outside'];lines.append(f"| {frame} | {s.get('estimate','undefined')} | {s.get('ci',s.get('reason','undefined'))} |")
 lines+=['','Positive inside-P oznacza pogorszenie przewidywania efektu interwencji dla badanego odczytu; ujemne oznacza poprawe. Nie jest to ocena jakosci naturalnych odpowiedzi.','', '## Wszystkie zachowane pary FIT','', '| Rama | FIT | P MSE | inside MSE | outside MSE |','|---|---|---:|---:|---:|']
 for frame in ['T5','T2']:
  for fit,v in result['frames'][frame]['per_fit'].items():lines.append('| '+frame+' | '+fit+' | '+' | '.join(f"{v['losses'][a]:.9g}" for a in ['P','inside','outside'])+' |')
 lines+=['','## Zakres i ograniczenia','', 'Oracle inside/outside uzywa prawdziwego D; nie jest metoda bez D. Kierunek U0 i mapy P sa ustalone z trzech wczesniejszych FIT, bez strojenia na nowych odpowiedziach. Przedzialy bootstrap-t95% z1999 zamrozonych losowan sa nominalne, bez nowej gwarancji pokrycia/FWER/mocy; nie raportujemy p ani odrzucenia rodziny. Brak wnioskowania o zmiennosci nowych refitow.','', 'To ograniczony kontrast rdzenia E041, nie dokladna replikacja historycznego B5/F1: inny model/operator/kohorta. Stare E041 oraz C3 pozostaja oddzielnymi wynikami. Dane T2 i T5 wspoldziela256blokow; pary FIT i warunki nie sa niezaleznymi probami.','', 'Wszystkie wyniki, energie reszt wektorowych, kontrasty inside-outside i dane blokowe pozostaja w analysis_result.json; nic nie wybrano na podstawie znaku. Odbior niezalezny: status sprawdzic w rzeczywistym completion_review; raport nie zastepuje receipt.']
 summary['outside_budget_scope']='Outside jest dopelnieniem U0:632 wymiary wobec8 wewnatrz U0 przy hidden640. Obie naprawy znaja true D, lecz nie sa dopasowane pod wzgledem liczby poprawianych kierunkow ani normy korekty. Nizsze MSE outside nie jest dowodem wiekszej efektywnosci na wymiar ani uniwersalnej przewagi dopelnienia.'
 put(BASE/'PRIMARY_SUMMARY.json',summary)
 with (BASE/'REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n\n'+summary['outside_budget_scope']+'\n')
if __name__=='__main__':main()
