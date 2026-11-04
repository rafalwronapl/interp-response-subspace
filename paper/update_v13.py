from pathlib import Path
import json, shutil, hashlib
p=Path(__file__).resolve().parent
root=p.parents[1]
# Earlier review artifacts are preserved solely as prior-version reviews.
if (p/'review').exists(): (p/'review').rename(p/'prior_v12_review')
t=(p/'main.tex').read_text(encoding='utf-8').replace('version 12','version 13').replace('This twelfth draft','This thirteenth draft')
t=t.replace('Thus useful correction does not establish an advantage over direct prediction.',r'A further fresh 64-block budget test finds 48.03\% lower MSE for correction than direct prediction at 32 labels, conditional on one fixed training sequence; Direct wins at 64--384 labels. Three new 32-label fits do not reproduce stable correction superiority: Direct is better on average for a new have/has readout, while the is/are mean interval crosses zero. Thus learned utility and comparative benefit are conditional.')
t=t.replace(r'\input{learned_results}',r'\input{learned_results}'+'\n'+r'\input{budget_transfer}')
t=t.replace('The two L21 TEST cohorts each contain 64 different pairs; only the 24 FIT blocks are reused in learned-model training.','The oracle and learned384 L21 TEST cohorts each contain 64 different pairs. The budget test adds a third 64-block cohort; the subsequent new-FIT/readout test reuses that budget TEST posthoc. The 24 historical FIT blocks are reused, and the latter test adds six new FIT blocks.')
t=t.replace('the two fresh 64-block L21 TEST studies have separate cohorts and estimands.','the oracle, learned384 and budget L21 TEST studies have separate cohorts; the subsequent new-FIT/readout test reuses the budget cohort posthoc.')
needle=r'\section{Relation to existing work}'
t=t.replace(needle,r'''The subsequent budget and new-FIT/readout studies sharpen that conclusion. A fixed 32-label C predictor beats Direct32 on fresh is/are TEST, whereas Direct has lower mean loss at every larger tested budget. Three disjoint new FIT32 cohorts then yield mixed rankings on the reused TEST: the new have/has readout favors Direct on average, and the is/are interval crosses zero. These results establish a conditional small-budget benefit, not stable superiority across FIT draws or readouts. Historical priors consume additional resources beyond the stated new-label budget.

'''+needle)
(p/'main.tex').write_text(t,encoding='utf-8')
d=(p/'discussion.tex').read_text(encoding='utf-8')
d=d.replace('Thus the result establishes scoped learned utility, not a superior residual-learning method.',r'The fresh budget TEST additionally identifies a conditional C32 advantage of 48.03\% over Direct32, but Direct wins at 64--384 labels and C64 worsens relative to C32. Three new FIT32 cohorts on the reused TEST favor Direct on average for have/has; the is/are mean interval includes zero. Thus the result establishes scoped learned utility and conditional comparative benefit, not stable superiority or a superior residual-learning method.')
d=d.replace('All conclusions concern one is/are margin in synthetic contexts,','Most conclusions concern one is/are margin in synthetic contexts; the posthoc have/has extension tests another grammatical margin on the same prompts,')
(p/'discussion.tex').write_text(d,encoding='utf-8')
sources=['analysis/AB_fast_followups_20261103/A_budget','analysis/AB_budget_readout_20261103','analysis/A_transfer_next_20261103']
manifest=[]
for src in sources:
 s=root/src; dest=p/'evidence'/'v13'/s.name
 wanted=['REPORT.txt','PROTOCOL.txt','completion_contract.json','completion_snapshot.json','completion_snapshot_v3.json','completion_manifest.json','input_manifest.json','canonical_verify.log','completion_checks.json','analysis_summary.json','run/summary.json','run/SUMMARY.json','run/completed.json']
 wanted += [str(f.relative_to(s)) for f in (s/'review').glob('*.json')]
 wanted += [str(f.relative_to(s)) for f in (s/'run').glob('*canonical*') if f.is_file()]
 for rel in dict.fromkeys(wanted):
  f=s/rel
  if not f.is_file():continue
  out=dest/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,out)
  manifest.append(dict(source=str(f.relative_to(root)).replace('\\','/'),copy=str(out.relative_to(p)).replace('\\','/'),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size))
(p/'v13_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(p/'CHANGELOG.txt').write_text('A13, November 3, 2026. Adds approved development vector budget curve, fresh native-readout budget TEST, and posthoc three-new-FIT/new-readout followup. Preserves all earlier results and byte-exact evidence. Corrects interpretation of new zero-MLP-delta arm without changing historical natural-margin zero-change arm. Prior reviews apply only to their original snapshots; A13 independent editorial review pending. No scientific calls, refits, resampling or new statistics.\n')
