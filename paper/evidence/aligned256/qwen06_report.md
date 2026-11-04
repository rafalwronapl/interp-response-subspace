# Oryginalny Qwen0.6B: naprawa wewnątrz i poza U oraz jej prognoza

Qwen3-0.6B-Base, oryginalny checkpoint E041 (da87bfb…/cd2a512…). Nowy ostatni blok L27, odczyt is–are. 256 nowych wspólnych bloków T2/T5, cztery warunki na ramę, trzy stałe pary FIT_MAP128/FIT_U128. Blok jest jednostką; pary FIT nie zwiększają niezależnego N.

| Rama | P MSE | inside oracle | outside oracle | inside−P | nominalny 95% CI |
|---|---:|---:|---:|---:|---|
| T5 | 0.490826902 | 0.535731597 | 0.108358087 | 0.044904695 | [0.013197211775560776, 0.07680531674126956] |
| T2 | 0.672180820 | 0.532339735 | 0.066455342 | -0.139841085 | [-0.17742600104380246, -0.10700736491593707] |

Przy tych samych trzech dopasowaniach naprawa inside średnio szkodzi na T5, a pomaga na T2. To przewidywanie efektu interwencji, nie ocena jakości naturalnych odpowiedzi. Outside ma niższe pooled MSE na obu ramach; obejmuje jednak 1016 kierunków wobec 8 inside, a normy korekt nie są dopasowane. Nie wynika stąd efektywność na wymiar ani uniwersalna przewaga dopełnienia.

| Rama | uczona projekcja | z selektorem | pełna uczona korekta | pooled direct256 | oracle znak | bezD znak | lepsza stała reguła znaku |
|---|---:|---:|---:|---:|---:|---:|---:|
| T5 | 0.566920743 | 0.521107066 | 0.444232088 | 0.379464250 | 99.479% | 64.746% | 63.249% |
| T2 | 0.608098401 | 0.594775159 | 0.535158287 | 0.505563005 | 99.837% | 68.620% | 59.375% |

BezD prognoza znaku przewyższa lepszą stałą regułę na obu ramach w tej próbie; pokrycie wynosi100%, remisów nie stwierdzono. To aktualizuje wcześniejszą ograniczoną obserwację czterech modeli, nie ustanawia uniwersalnego selektora. Selektor poprawia średni wynik względem zawsze stosowanej projekcji na obu ramach, ale na T5 nadal szkodzi względem P; na obu ramach lepszy jest pooled direct256. Trafność znaku i użyteczność są osobnymi wynikami.

Oracle zna prawdziwe D i używa innej korekty niż wariant bezD; różnica trafności nie izoluje samego usunięcia D. Wariant bezD nadal potrzebuje natywnych stanów odbiorcy/wejść i gradientu. Wszystkie dziewięć krzyżowanych par, wszystkie ujemne wyniki i szczegóły dopasowań zachowano w summary.json; trzy matched pary są zakresem głównym.

## Relacja do E041

To ten sam model bazowy, ale E041 miało L21, inny B5/F1, dane i dopasowania. Nie jest to identyczna replikacja wyniku98,85% inside /14,44% outside. Nowe wyniki pokazują warunkowość naprawy w innym kontrolowanym układzie; nie pozwalają przypisać różnicy wyłącznie warstwie, danym albo architekturze. Wybór rank8U pozostaje wektorowy, bez bezpośrednich etykiet odczytu.

## Analiza, koszt i dowody

Wspólny zamrożony analizator A:1999draw, nominalne bootstrap-t95% CI, bez gwarancji pokrycia/FWER ani wnioskowania po populacji refitów. C podaje opisowe średnie i jawne konwencje remisów/pokrycia; nie powtarzano bootstrapu dla tych samych kontrastów.

Capture:2844F (2816naturalnych+24pełne kontrole+4suffix),0B/1pełne załadowanie. A/C:102400tanich norm/headF,0B/0pełnych modeli;6144unikalne punkty gradientu,18432rzeczywiste ewaluacje analityczne przy regeneracji.2048bundle receipts zawierają wszystkie102400logicznych ID; nie usunięto ramion. V1 zatrzymało się na błędzie pola cohort przed pierwszym FIT/odczytem (0F/0B), zachowano logi; literalny fix role-before-cohort dostał osobny snapshot/recenzję i dane nie były ponownie mierzone.

Wyniki: evaluation_v2/run/result.json; summary.json; block_summary.json. Rzeczywisty odbiór: evaluation_v2/completion_review_sol_gemma_peer_v1.json — status czytać z receipt, ten raport sam go nie zastępuje. B jest osobnym następnym etapem, nie ukończonym wynikiem.

Semantyka energii: surowe C energy_inside=||c||² i energy_outside=||e−c||² to składowe pierwotnego błędu usuwane przez odpowiednie naprawy. W aligned_A.vector_energies zapisano błąd PO naprawie: inside pozostawia ||e−c||², outside pozostawia ||c||². Translator zamienia te etykiety świadomie; wartości i pomiary nie są zmieniane.
