# Gemma270: warunkowosc naprawy podprzestrzennej

256 nowych blokow leksykalnych, T2/T5, cztery warunki/rame i trzy zachowane pary P/U0. Bez nowego FIT. Jednostka analizy: blok; trzy FIT i cztery warunki usredniono wewnatrz bloku. Primary: T5 inside-P. T2 oraz inside-outside pozostaja opisowe.

| Rama | P MSE | inside MSE | outside MSE | inside-P | nominal95% CI |
|---|---:|---:|---:|---:|---|
| T5 | 0.449096903 | 0.332635133 | 0.164258692 | -0.11646176918701083 | [-0.14403551848414717, -0.09173171271059781] |
| T2 | 0.209279167 | 0.0832558775 | 0.05597879 | -0.12602328968582496 | [-0.140182100419433, -0.11384555781222463] |

## Inside-outside: kontrast opisowy

| Rama | inside-outside | nominal95% CI |
|---|---:|---|
| T5 | 0.168376441287745 | [0.13536037704168777, 0.2078074494140332] |
| T2 | 0.027277087487939333 | [0.019143126405648227, 0.03623523697096661] |

Positive inside-P oznacza pogorszenie przewidywania efektu interwencji dla badanego odczytu; ujemne oznacza poprawe. Nie jest to ocena jakosci naturalnych odpowiedzi.

## Wszystkie zachowane pary FIT

| Rama | FIT | P MSE | inside MSE | outside MSE |
|---|---|---:|---:|---:|
| T5 | r1 | 0.349193618 | 0.314873815 | 0.0615041127 |
| T5 | r2 | 0.356220917 | 0.188821413 | 0.19682743 |
| T5 | r3 | 0.641876173 | 0.494210172 | 0.234444534 |
| T2 | r1 | 0.220346403 | 0.0919347956 | 0.0491978475 |
| T2 | r2 | 0.219165966 | 0.0682359482 | 0.0824162927 |
| T2 | r3 | 0.188325133 | 0.0895968887 | 0.03632223 |

## Zakres i ograniczenia

Oracle inside/outside uzywa prawdziwego D; nie jest metoda bez D. Kierunek U0 i mapy P sa ustalone z trzech wczesniejszych FIT, bez strojenia na nowych odpowiedziach. Przedzialy bootstrap-t95% z1999 zamrozonych losowan sa nominalne, bez nowej gwarancji pokrycia/FWER/mocy; nie raportujemy p ani odrzucenia rodziny. Brak wnioskowania o zmiennosci nowych refitow.

To ograniczony kontrast rdzenia E041, nie dokladna replikacja historycznego B5/F1: inny model/operator/kohorta. Stare E041 oraz C3 pozostaja oddzielnymi wynikami. Dane T2 i T5 wspoldziela256blokow; pary FIT i warunki nie sa niezaleznymi probami.

Wszystkie wyniki, energie reszt wektorowych, kontrasty inside-outside i dane blokowe pozostaja w analysis_result.json; nic nie wybrano na podstawie znaku. Odbior niezalezny: status sprawdzic w rzeczywistym completion_review; raport nie zastepuje receipt.

Outside jest dopelnieniem U0:632 wymiary wobec8 wewnatrz U0 przy hidden640. Obie naprawy znaja true D, lecz nie sa dopasowane pod wzgledem liczby poprawianych kierunkow ani normy korekty. Nizsze MSE outside nie jest dowodem wiekszej efektywnosci na wymiar ani uniwersalnej przewagi dopelnienia.
