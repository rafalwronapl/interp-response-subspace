# L21 learned correction: FIT-only feasibility

Nested lexical-block cross-validation on24existingFITblocks/384nativevectorlabels. Historical frozenP/U is extra prior training. Eachouterfold trains256 andvalidates128labels. Zero newnative/suffix evaluations. This does NOT measure downstream readout utility.

| Method | Vector squared L2 | Coefficient squared L2 |
|---|---:|---:|
| C | 3.20825427 | 0.0478473796 |
| P | 4.30269555 | 1.14228866 |
| direct | 0.414476817 | 0.0448341419 |
| mean | 4.28882321 | 1.12841632 |
| same_U | 3.21181039 | 0.0514034969 |

C=P+U*learnedcoefficients; same_U is direct linear rank8refit; direct predictsfullD with independently selected linear/RBF model. Allmethods use same newtraininglabels. Models finalizedon384FITlabels, no futureTESTnative targets used.

No claim of learned readout improvement can be made before separate newholdout/native suffix measurements. Current64blockA TEST alreadyexposed, so upcoming stage2 uses new64blocks. Nominal futureblockintervals notindependentlyfamilyqualified.

Final chosen hyperparameters: {"C": {"cv": [{"block_L2": 0.4580446237696581, "kernel": "linear", "lam": 1.0}, {"block_L2": 0.13388530103207777, "kernel": "linear", "lam": 0.01}, {"block_L2": 0.04771349525358647, "kernel": "linear", "lam": 0.0001}, {"block_L2": 0.03995622036534336, "kernel": "linear", "lam": 1e-06}, {"block_L2": 0.7025816990030099, "kernel": "rbf", "lam": 1.0}, {"block_L2": 0.13332373809424466, "kernel": "rbf", "lam": 0.01}, {"block_L2": 0.049948826830610187, "kernel": "rbf", "lam": 0.0001}, {"block_L2": 0.04512451973807722, "kernel": "rbf", "lam": 1e-06}], "param": {"kernel": "linear", "lam": 1e-06}}, "direct": {"cv": [{"block_L2": 2.773423116735117, "kernel": "linear", "lam": 1.0}, {"block_L2": 0.9619673093038736, "kernel": "linear", "lam": 0.01}, {"block_L2": 0.47426029738470366, "kernel": "linear", "lam": 0.0001}, {"block_L2": 0.41235660430324456, "kernel": "linear", "lam": 1e-06}, {"block_L2": 4.196297089896309, "kernel": "rbf", "lam": 1.0}, {"block_L2": 0.9624726145370731, "kernel": "rbf", "lam": 0.01}, {"block_L2": 0.4828639497089244, "kernel": "rbf", "lam": 0.0001}, {"block_L2": 0.44005600274217077, "kernel": "rbf", "lam": 1e-06}], "param": {"kernel": "linear", "lam": 1e-06}}, "same_U": {"cv": [{"block_L2": 1.2690367496514634, "kernel": "linear", "lam": 1.0}, {"block_L2": 0.23158502838408335, "kernel": "linear", "lam": 0.01}, {"block_L2": 0.05594759441141212, "kernel": "linear", "lam": 0.0001}, {"block_L2": 0.041923903704339356, "kernel": "linear", "lam": 1e-06}], "param": {"kernel": "linear", "lam": 1e-06}}}
Independent completion status is recorded separately in completion_review.json.
