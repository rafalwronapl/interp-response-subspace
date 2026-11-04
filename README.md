# An Intervention-Response Subspace Retains Nearly All Readout-Relevant Prediction Error: A Case Study in Qwen3-0.6B

Author: Rafał Wrona (independent researcher)

A rank-8 space fitted from native responses to an internal intervention in Qwen3-0.6B-Base (layer-19 SwiGLU neuron 1317, target layer-21 MLP output), without readout labels, retains almost all of the is/are-readout-relevant error of a vector predictor: oracle repair inside it reduces readout MSE by 99.25% on fresh contexts. Controls: principal subspaces of natural activations, response spaces of three unscreened neurons (also at matched strength), 20 random spaces and a gradient ceiling; learned-correction comparisons including a prior-matched direct predictor.

## Contents

- `paper/` – LaTeX source, figures, compiled `main.pdf` (article with appendix) and `supplement.pdf` (full extended record of all earlier stages, failed forecasts, operation accounting).
- `paper/evidence*`, `paper/*_evidence/` – compact evidence copied for the manuscript: frozen protocols, summaries, independent review receipts, source hashes.
- `experiments/` – code, protocols, reports, summaries, configs and reviews of each experiment stage:
- `experiments/analysis__AB_five_experiments_20261103__A_L21` (lab path `analysis/AB_five_experiments_20261103/A_L21`)
- `experiments/analysis__A_L21_learned_20261103` (lab path `analysis/A_L21_learned_20261103`)
- `experiments/analysis__AB_budget_readout_20261103` (lab path `analysis/AB_budget_readout_20261103`)
- `experiments/analysis__A_transfer_next_20261103` (lab path `analysis/A_transfer_next_20261103`)
- `experiments/analysis__A_prior_matched_20261104` (lab path `analysis/A_prior_matched_20261104`)
- `experiments/analysis__A_U_gradient_pca_20261104` (lab path `analysis/A_U_gradient_pca_20261104`)
- `experiments/analysis__A_gemma270_extension_20261102` (lab path `analysis/A_gemma270_extension_20261102`)
- `experiments/analysis__Qwen06_ABC_extension_20261102` (lab path `analysis/Qwen06_ABC_extension_20261102`)
  Per-condition result rows of a stage are merged into `rows.jsonl.gz` (one JSON object per original row file). Files larger than 5 MB (frozen forecasts, fitted models, schedules) are stored as `.gz`.
- `lab_code/` – shared lab modules imported by the experiment workers (model loading, native hooks, tensor encoding, launch/review gate).

## Not included

Model weights (public Hugging Face checkpoints: Qwen/Qwen3-0.6B-Base, google/gemma-3-270m, google/gemma-3-1b-pt, Gemma Scope 2 transcoders, BlueLightAI Qwen3-0.6B CLT), raw activation caches, per-call ledgers and intermediate bundles (several GB). Two frozen forecast files exceed GitHub limits even compressed; their SHA-256 hashes are recorded in the stage manifests.

## Reproducing tables

Every table in `paper/main.tex` reproduces values from the `summary.json` / `REPORT` files of the corresponding stage; no numbers in the manuscript are recomputed at writing time. Re-running a stage requires the model checkpoints and the lab environment (Python 3, PyTorch, transformers); workers expect the original lab layout (`lab_code/` at the repository root of the lab).

Paths inside configs refer to the author's local lab layout and are kept unchanged for provenance.

## License

Code: MIT. Text, figures and data: CC BY 4.0.
