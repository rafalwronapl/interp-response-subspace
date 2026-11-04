# Qwen0.6B: publiczny komponent CLT i wielkość interwencji

Oryginalny Qwen3-0.6B-Base da87bfb…/cd2a512…; publiczny BlueLightAI CLT20k, source27→target27, rewizja5301dca…. Ostatni MLP L27 i odczyt is–are. Jedyny operator: pełny kierunek normalized-MLP-input, x_c=x+c(x_d−x). 256 bloków wspólnych z A/C, T2/T5, cztery warunki na ramę, sześć dawek.

To opisowy follow-up na wspólnej kohorcie, nie niezależne świeże potwierdzenie GemmaB. Native/RIDGE przyc1 były znane z A/C i zostały adoptowane (4096odczytów). Nowe odpowiedzi CLT/Taylora/FROZEN oraz pozostałych dawek nie były mierzone przed zamrożeniem B; wybrano pierwszy z góry ustalony FTcohort r1, bez wyboru najlepszego dopasowania po A/C.

Główny TAYLOR oznacza natywny wektorowy JVP MLP, skalowany dawką i propagowany przez dokładny nieliniowy norm/head, tak jak w oryginalnym GemmaB. COMPOSED_LINEAR to odrębna diagnostyka: skalarny Taylor całego odczytu. FROZEN ma maskę encoderu z recipienta; natywna gałąź i finalnorm/head pozostają rzeczywiste. Ridge i średnia warunkowa side/q wykorzystują wyłącznie128FT_MAPr1odpowiedzi wektorowych, lambda0,01; unitridge fit adoptowano z A/C.

## T5: błąd względny RMS

To RMS błędu odczytu podzielony przez RMS natywnego efektu w tej samej komórce; nie odsetek złych odpowiedzi ani procent wyjaśnionego modelu.

| dawka | native RMS | CLT | TAYLOR | COMPOSED_LINEAR | FROZEN | RIDGE | MEAN | ZERO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.1492235 | 99.712% | 3.942% | 4.095% | 99.693% | 35.781% | 38.394% | 100.000% |
| 0.1 | 0.2903733 | 99.670% | 7.431% | 7.746% | 99.652% | 36.548% | 39.292% | 100.000% |
| 0.25 | 0.6821087 | 99.638% | 15.242% | 16.056% | 99.628% | 37.086% | 41.361% | 100.000% |
| 0.5 | 1.284321 | 99.877% | 24.245% | 25.721% | 99.851% | 33.394% | 42.078% | 100.000% |
| 1.0 | 2.538302 | 100.625% | 42.349% | 44.003% | 100.549% | 25.248% | 38.590% | 100.000% |
| 2.0 | 8.85417 | 102.614% | 53.576% | 49.535% | 100.484% | 24.526% | 32.247% | 100.000% |

### T5: bezwzględny RMS błędu odczytu

| dawka | CLT | TAYLOR | COMPOSED_LINEAR | FROZEN | RIDGE | MEAN | ZERO |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.1487937 | 0.005882105 | 0.006111284 | 0.1487658 | 0.05339326 | 0.0572928 | 0.1492235 |
| 0.1 | 0.2894142 | 0.02157897 | 0.02249287 | 0.2893635 | 0.1061245 | 0.1140948 | 0.2903733 |
| 0.25 | 0.6796383 | 0.1039674 | 0.109522 | 0.6795735 | 0.2529644 | 0.28213 | 0.6821087 |
| 0.5 | 1.282746 | 0.3113851 | 0.3303446 | 1.282406 | 0.4288897 | 0.540422 | 1.284321 |
| 1.0 | 2.554165 | 1.074939 | 1.116926 | 2.552248 | 0.6408734 | 0.9795418 | 2.538302 |
| 2.0 | 9.085619 | 4.743741 | 4.385956 | 8.897025 | 2.171576 | 2.855191 | 8.85417 |

## T2: błąd względny RMS

To RMS błędu odczytu podzielony przez RMS natywnego efektu w tej samej komórce; nie odsetek złych odpowiedzi ani procent wyjaśnionego modelu.

| dawka | native RMS | CLT | TAYLOR | COMPOSED_LINEAR | FROZEN | RIDGE | MEAN | ZERO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.1211807 | 100.009% | 1.326% | 1.433% | 100.036% | 37.562% | 36.334% | 100.000% |
| 0.1 | 0.240322 | 100.010% | 2.513% | 2.720% | 100.032% | 37.989% | 36.797% | 100.000% |
| 0.25 | 0.590596 | 100.028% | 5.706% | 6.154% | 100.024% | 38.857% | 38.109% | 100.000% |
| 0.5 | 1.166802 | 100.022% | 11.079% | 11.743% | 100.019% | 38.905% | 39.119% | 100.000% |
| 1.0 | 2.351438 | 100.022% | 22.414% | 23.129% | 100.014% | 35.790% | 36.762% | 100.000% |
| 2.0 | 6.494448 | 99.977% | 40.512% | 36.932% | 99.958% | 22.037% | 27.496% | 100.000% |

### T2: bezwzględny RMS błędu odczytu

| dawka | CLT | TAYLOR | COMPOSED_LINEAR | FROZEN | RIDGE | MEAN | ZERO |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.1211919 | 0.001606616 | 0.001736221 | 0.1212241 | 0.04551736 | 0.04403023 | 0.1211807 |
| 0.1 | 0.2403465 | 0.0060386 | 0.006537385 | 0.2403993 | 0.09129553 | 0.08843019 | 0.240322 |
| 0.25 | 0.5907585 | 0.0336972 | 0.03634236 | 0.5907406 | 0.229489 | 0.2250714 | 0.590596 |
| 0.5 | 1.167054 | 0.1292736 | 0.1370139 | 1.167028 | 0.4539403 | 0.456445 | 1.166802 |
| 1.0 | 2.351965 | 0.5270398 | 0.5438677 | 2.351763 | 0.8415912 | 0.8644464 | 2.351438 |
| 2.0 | 6.492923 | 2.63103 | 2.398533 | 6.491713 | 1.431181 | 1.785718 | 6.494448 |

## mixture: błąd względny RMS

To RMS błędu odczytu podzielony przez RMS natywnego efektu w tej samej komórce; nie odsetek złych odpowiedzi ani procent wyjaśnionego modelu.

| dawka | native RMS | CLT | TAYLOR | COMPOSED_LINEAR | FROZEN | RIDGE | MEAN | ZERO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.1359272 | 99.830% | 3.172% | 3.305% | 99.830% | 36.499% | 37.589% | 100.000% |
| 0.1 | 0.2665251 | 99.808% | 5.945% | 6.214% | 99.807% | 37.140% | 38.297% | 100.000% |
| 0.25 | 0.6379953 | 99.805% | 12.113% | 12.789% | 99.798% | 37.855% | 40.000% | 100.000% |
| 0.5 | 1.226969 | 99.943% | 19.430% | 20.610% | 99.927% | 35.991% | 40.767% | 100.000% |
| 1.0 | 2.446654 | 100.347% | 34.600% | 35.904% | 100.302% | 30.572% | 37.757% | 100.000% |
| 2.0 | 7.764476 | 101.699% | 49.401% | 45.525% | 100.300% | 23.685% | 30.669% | 100.000% |

### mixture: bezwzględny RMS błędu odczytu

| dawka | CLT | TAYLOR | COMPOSED_LINEAR | FROZEN | RIDGE | MEAN | ZERO |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.1356964 | 0.004311634 | 0.004492341 | 0.1356955 | 0.04961184 | 0.05109366 | 0.1359272 |
| 0.1 | 0.2660141 | 0.01584482 | 0.01656301 | 0.2660104 | 0.0989881 | 0.1020724 | 0.2665251 |
| 0.25 | 0.636751 | 0.07728103 | 0.08159606 | 0.6367081 | 0.2415121 | 0.2552004 | 0.6379953 |
| 0.5 | 1.226265 | 0.2384034 | 0.2528837 | 1.226075 | 0.4415927 | 0.500199 | 1.226969 |
| 1.0 | 2.455148 | 0.8465414 | 0.8784404 | 2.454053 | 0.7479955 | 0.9237883 | 2.446654 |
| 2.0 | 7.896408 | 3.835713 | 3.534796 | 7.787791 | 1.839024 | 2.381271 | 7.764476 |

## Zakres i dowody

Wszystkie126komórek oraz błędy perblok pozostają w B_v1/summary.json i block_summary.json; nie wybierano dodatnich dawek ani ram. Oba szablony i metody współdzielą256bloków. Brak kwalifikowanych CI/FWER, p-values lub gwarancji populacyjnej. Niższy błąd względny przy większej dawce należy interpretować razem z błędem absolutnym i natywnym sygnałem; wspólny mianownik nie tłumaczy różnicy rankingu metod w tej samej komórce.

To lokalny komponent publicznego CLT, nie całe replacement network ani dokładna replikacja GemmaScopeTC/L15/she–he. Wcześniejsze źródła CLT są ustalone przez lokalną interwencję przyL27; b_dec anuluje się w różnicy. Przenosimy różnicę cech przez decoder wFP32. Encoder baseline i endpoint używają tego samego serialnego kształtu kernela, aby zmiana batchshape nie tworzyła sztucznego przejścia przez próg.

Koszt nowy96924logicznychheterogenicznychF:10896lokalnych gałęziMLP,69644odczytynorm/head,12288predykcjiendpointCLT,2048bazowychkodowańCLT,2048kierunkówFROZEN.0B i0pełnych załadowań LLM; czytano częściowe wagi nativeMLP/publicCLT.2052punkty z analitycznym natywnymJVP i gradientem odczytu podano oddzielnie.4096znanychunitheadów nie doliczono drugi raz. Wszystkie4kontrole branch/head/JVP i niezerowej osiągalności przeszły przed TEST.

Rzeczywisty odbiór: B_v1/completion_review_sol_gemma_peer_v1.json — status czytać z receipt. Raport sam nie nadaje aprobaty ani nie ustanawia światowego pierwszeństwa.
