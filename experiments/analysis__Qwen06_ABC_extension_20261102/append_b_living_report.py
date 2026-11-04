"""Append independently accepted B stage to the stable user report once."""
from pathlib import Path
from datetime import datetime, timezone
from analysis.C2_fresh_fit_holdout_20261102.worker import read
BASE=Path(__file__).parent
review=read(BASE/'B_v1/completion_review_sol_gemma_peer_v1.json')
assert review['verdict'].lower()=='approved', review
target=Path('C:/Users/rafal/Downloads/Dorobek_programu_badawczego_2026-10-22.md')
marker='## 2026-11-02: Qwen0.6 B — zakończony lokalny test CLT27'
text=target.read_text(encoding='utf-8')
assert marker not in text, 'Already recorded; do not append twice'
addition=f'''

{marker}

Stan wpisu: {datetime.now(timezone.utc).isoformat()}. Wszystkie 96924 nowe logiczne operacje zakończone, runtime e8c33311551c4abfbba77263354a26e8 exit0. Oryginalny Qwen3-0.6B-Base da87…/cd2a512…, ostatni MLP L27, odczyt is–are; publiczny CLT BlueLightAI20k rev5301… wyłącznie komponent source27→target27. Pełny kierunek wejścia MLP, sześć dawek .05/.1/.25/.5/1/2, 256 wspólnych z A/C bloków, 2048 kontekstów T2/T5, siedem metod, komplet126komórek. To opisowy follow-up: 4096 znanych z A/C odczytów native/ridge przyc1 adoptowano bez ponownego liczenia; nie jest to niezależne świeże potwierdzenie B. Nowe wyniki CLT/Taylor/FROZEN oraz pozostałych dawek nie były znane przed zamrożeniem B.

Wynik ujemny dla postulowanego użytecznego dużego reżimu tego komponentu: względny RMS CLT dla mieszanki T2/T5 wynosi99.83% przyc=.05,100.35% przyc=1 i101.70% przyc=2 (ZERO=100%). Natywny wektorowy Taylor z dokładnym nieliniowym headem:3.17%/34.60%/49.40%; ridge przyc1:30.57%, przyc2:23.69%. Pełna tabela zawiera też błędy absolutne, natywny sygnał, obie ramy oddzielnie, frozen-mask/mean oraz oddzielny scalar COMPOSED_LINEAR. Nie wybierano dodatnich dawek/ram; brak nowej kalibracji CI/FWER/p. Nie jest to globalna ocena pełnej sieci CLT ani identyczna replikacja GemmaB L15/she–he; nie obala dodatniego wyniku GemmaB. Lokalne źródła oraz oficjalne encode/loader potwierdzają deklarowaną granicę mlp.hook_in/out; kontrole native branch/head/JVP przeszły. Nie ustanawia to uniwersalnej jakości zewnętrznego artefaktu.

Nowy koszt:10896lokalnychMLP +69644norm/head +12288CLTendpoint +2048CLTbaseline +2048frozen-directions=96924heterogenicznychF,0B,0pełnychLLMloads;2052punkty analitycznegoJVP/gradientu oddzielnie. Cały nowy Q A/C/B:202168heterogenicznychF (2844capture+102400A/C+96924B),0B,1pełneLLMload. Wszystko naC; stareE041/L21 zachowano osobno. Niezależny odbiór APPROVED: analysis/Qwen06_ABC_extension_20261102/B_v1/completion_review_sol_gemma_peer_v1.json. Dowody: analysis/Qwen06_ABC_extension_20261102/B_REPORT.md, B_PRIMARY_SUMMARY.json, B_v1/{{summary.json,block_summary.json,completion_snapshot_v1.json}}. Szczegółowy zakres i ograniczenia niezależnego audytu są w receipt; bez powtórnych pomiarów/bootstrap. Ten wpis zastępuje status bieżący wcześniejszego4.52%checkpointu, pozostawionego historycznie.
'''
target.write_text(text+addition,encoding='utf-8')
print('Accepted B outcome appended once to stable report')
