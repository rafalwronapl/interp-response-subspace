"""Summarize scoped results without converting prediction into semantic discovery."""
from .common import OUT,payload,events

def main():
    result=payload(OUT/'test/results.json');fit=payload(OUT/'development/predictors.json')['selection']
    lines=['E036 — WYNIK LOKALNEGO TESTU PRZEWIDYWANIA RESZTY CLT',
        'Decyzja H1: '+result['decision'],
        '224 prompty główne: FIT96 / SELECT32 / TEST96; TEST12 niezależnych bloków,384 wektory.',
        'To lokalna prognoza MLP21 przy prawdziwych wejściach po ingerencji; nie pełny symulator ani etykieta semantyczna.',
        'Wybrany kandydat: '+str(fit['candidate']), '', 'Błędy pełnych wektorów:']
    for k,v in result['metrics'].items():lines.append(k+': '+str(v))
    lines+=['','Wszystkie obowiązkowe porównania H1:']+[k+': '+str(v) for k,v in result['comparisons'].items()]
    lines+=['','Dodatkowa wartość atlasu vs bezpośrednia mapa: '+str(result['additional_CLT_value']),
        'Mała ranga vs pełna ridge: '+str(result['low_rank_vs_full']),
        'Poprawa wektorów nie potwierdza poprawy odpowiedzi is/are; wymaga osobnego testu przyczynowego.',
        'Ujemny wynik ogranicza tę rodzinę korekt, te dawki i ramy; nie dowodzi braku przewidywalnej struktury w ogóle.',
        'Wynik wymaga niezależnej końcowej recenzji; pełne dane: test/results.json i surowe receipts.', '', 'Koszt:']
    for stage in ('pilot','development','test'):lines.append(stage+': '+str(payload(OUT/stage/'complete.json')['ledger']))
    text='\n'.join(lines)+'\n';path=OUT/'PODSUMOWANIE.txt'
    if path.exists() and path.read_text(encoding='utf-8')!=text:raise ValueError('Report changed')
    path.write_text(text,encoding='utf-8');print(text)

if __name__=='__main__':main()
