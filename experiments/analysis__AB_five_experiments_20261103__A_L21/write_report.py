"""Render existing frozen aggregates; no fitting, model calls or resampling."""
from pathlib import Path
import json
BASE=Path(__file__).parent
def main():
 s=json.loads((BASE/'main_run/summary.json').read_text())['payload'];m=s['metrics'];base=sum(m[f'f{j}_P']['MSE'] for j in range(3))/3
 lines=['A experiments2/4: completed fresh E041/L21 replication and separate rank controls','2026-11-03; scientific outcome report, independent completion review separate.','',
 'Scope: Qwen3-0.6B-Base original checkpoint FP32; native L19 neuron1317 intervention, branchB free L20 or branchC restores neuron1423, targetL21; doses0.5/1, F1 is/are readout. 64 fresh lexical-pair TEST blocks x4 forms x2branches x2doses=1024 conditions. Historical B5/U frozen unchanged; no CLT. Three distinct new FIT cohorts8blocks each for separate rankstudy; same64TESTblocks shared acrossfits, not192 independent blocks.',
 'All native-delta forecasts frozen globally before any TEST target. Subspaces from vector FIT labels, not TEST readout labels. Corrections remain ORACLE: true TEST delta/error required.','',
 'EXPERIMENT2: exact historical fit, primary aggregation across all prespecified branches/doses.']
 for a in ['old_P','old_inside','old_outside','old_oracle']:
  lines.append(f"{a}: MSE={m[a]['MSE']:.12g}; nominal95 MSE interval={m[a]['nominal95']}")
 for a,v in s['primary'].items():lines.append(f"{a}: reduction={100*v['reduction']:.6f}%; paired deltaMSE={v['delta_MSE']:.12g}; nominal95 delta interval={v['nominal95']}")
 lines+=['Historical exploratory inside reduction98.85% is qualitatively and approximately quantitatively reproduced here (99.2454%). Outside historical14.44% versus current8.3805%; do not claim identical outside effect. This is an independent fresh-pair test conditional on the historically selected checkpoint/operator/readout/fit, not an independent choice of model/task.','Subgroups (descriptive):']
 for tag,v in s['subgroups'].items():lines.append(f"{tag}: baseline={v['old_P']:.12g}, inside={v['old_inside']:.12g}, outside={v['old_outside']:.12g}; reductions inside={100*(1-v['old_inside']/v['old_P']):.4f}%, outside={100*(1-v['old_outside']/v['old_P']):.4f}%")
 lines+=['','EXPERIMENT4: separate new fits; equal-fit mean MSE, same64blocks.','New-fit baseline mean MSE='+format(base,'.12g'),'rank | FIT-derived projection MSE | random same-rank MSE | random norm-rescaled MSE | FIT-derived reduction%']
 for r in [1,2,4,8,16,32]:
  vals=[sum(m[f'f{j}_r{r}_{k}']['MSE'] for j in range(3))/3 for k in ['learned','random','random_norm']]
  lines.append(f"{r} | {vals[0]:.12g} | {vals[1]:.12g} | {vals[2]:.12g} | {100*(1-vals[0]/base):.6f}")
 lines+=['All rankstudy deltas and precomputed nominal paired intervals versus their own newFIT baseline:']
 for r in [1,2,4,8,16,32]:
  for k in ['learned','random','random_norm']:
   v=s['rank_summary'][f'r{r}_{k}'];lines.append(f"r{r}/{k}: delta={v['delta_MSE_equal_fit']:.12g}; nominal95={v['nominal95']}")
 lines+=['','Interpretation: supports a task-relevant low-dimensional native-delta FIT structure for this exact L21/F1 setting. FIT-derived projection outperforms the chosen fixed random controls descriptively at every measured rank, including controls rescaled to the same correction norm. Rank8 is effective but not a uniquely optimal dimension: larger ranks further reduce MSE. The norm-rescaled random arm is a separate intervention, not exact orthogonal oracle projection. Comparisons are between the fixed chosen random draws and fitted subspaces, not all random subspaces.',
 'Limitations: oracle correction requires true test error; this does not establish learned correction utility or superiority to Direct256, does not generalize to other layers/readouts/models, and does not reverse prior L27 context-dependent/harmful findings. New-fit baselines differ from historicalfit and must not be merged into a single claimed replication. Nominal block bootstrap1999draws seed3110301, conditional onfixedfits, no familywise/multiplicity qualification or population-power claim. No new worldwide-priority claim. All prespecified ranks/controls and outcomes retained.',
 'Evidence: main_run/summary.json, rows/, ledger.jsonl, TEST_forecasts.json, fit_0/1/2.json, completed.json; main_review.json and parallel/review_v2.json for pre-run; completion_review.json will record independent outcome acceptance. No model calls/refits/resampling in this rendering.']
 (BASE/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('\n'.join(lines[:22]))
if __name__=='__main__':main()
