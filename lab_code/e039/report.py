from .common import OUT,payload

def main():
    pilot=payload(OUT/'pilot/checks.json')
    if not (OUT/'main/results.json').exists():
        text=f"E039 zakończone na pilocie. Gotowość: {pilot['ready']}.\nSygnał: {pilot['signal']}\n"
    else:
        r=payload(OUT/'main/results.json')
        text=f"E039 zakończone. H1_joint={r['H1_joint']}.\n"
        for frame,result in r['frames'].items():
            text+=f"F{frame}: B5_transfer={result['transfers']['B5']['decision']}; C_transfer={result['transfers']['C']['decision']}; B5_leads={result['B5_leads']['decision']}; C_leads={result['C_leads']['decision']}.\n"
            for k,v in result['metrics'].items():text+=f"  {k}: RMSE={v['RMSE']:.9g}, NRMSE_H={v['NRMSE_H']}\n"
        text+='Zakres: warunkowa prognoza zmiany MLP21 przy natywnym upstream/downstream; nie pełna rekonstrukcja obwodu.\n'
    with (OUT/'PODSUMOWANIE.txt').open('x',encoding='utf-8') as f:f.write(text)

if __name__=='__main__':main()
