from __future__ import annotations
import numpy as np
import pandas as pd
from DiagnoseHarmonization import DiagnosticFunctions
import warnings
warnings.filterwarnings("ignore")

 # Synthetic data: Generate test data
np.random.seed(42)

# design: 6 subjects, 2 timepoints each -> 12 rows
subjects = []
timepoints = []
batches = []
age = []
subj_ids = [f"S{i}" for i in range(1, 7)]
# assign subjects to batches (3 batches: A, B, C), two subjects per batch
subj_to_batch = {subj_ids[0]: "A", subj_ids[1]: "A",
                 subj_ids[2]: "B", subj_ids[3]: "B",
                 subj_ids[4]: "C", subj_ids[5]: "C"}

for s in subj_ids:
    for t in ("T1", "T2"):
        subjects.append(s)
        timepoints.append(t)
        batches.append(subj_to_batch[s])
        # give ages that vary by subject (but constant across that subject for simplicity)
        base_age = 30 + (int(s[1:]) * 3)  # e.g., S1 -> 33, S2 -> 36, ...
        age.append(float(base_age))

n_samples = len(subjects)  # should be 12
# Create two IDPs with:
# - between-batch offsets: batch A=+0.0, B=+1.5, C=+3.0
# - between-subject random effect (subject-specific offset)
# - within-subject noise small so ICC>0
batch_offsets = {"A": 0.0, "B": 1.5, "C": 3.0}
n_idps = 2
idp_matrix = np.zeros((n_samples, n_idps), dtype=float)

# subject-level random effects
subj_effect = {s: np.random.normal(loc=0.0, scale=0.6, size=n_idps) for s in subj_ids}

for i in range(n_samples):
    s = subjects[i]
    b = batches[i]
    # base signal per IDP
    base = np.array([2.0, 5.0])
    # batch effect (same for both IDPs but scaled differently)
    batch_eff = np.array([1.0, 0.5]) * batch_offsets[b]
    # subject-specific offset
    sub_eff = subj_effect[s]
    # timepoint effect (T2 slight increase)
    tp_eff = np.array([0.3, -0.2]) if timepoints[i] == "T2" else np.array([0.0, 0.0])
    # observation noise
    noise = np.random.normal(scale=0.2, size=n_idps)
    idp_matrix[i, :] = base + batch_eff + sub_eff + tp_eff + noise

idp_names = ["IDP_A", "IDP_B"]

# sanity checks (optional)
print("n_samples:", n_samples)
print("subjects (unique):", sorted(set(subjects)))
print("batches (unique):", sorted(set(batches)))
print("shape idp_matrix:", idp_matrix.shape)
print("first 5 rows of idp_matrix:\n", idp_matrix[:5])
print("Data generation completed") 

#####################################

# 1) Subject order
subjorder = DiagnosticFunctions.SubjectOrder_long(idp_matrix=idp_matrix,
    subjects=subjects,
    timepoints=timepoints,
    idp_names=idp_names,
    nPerm=1000)
print("\SUBJECT ORDER CONSISTENCY: RANK CORRELATIONS WITH PERMUTATION TESTS")
print(subjorder)

# 2) Within Subject Variability
wsv = DiagnosticFunctions.WithinSubjVar_long(
    idp_matrix=idp_matrix,
    subjects=subjects,
    timepoints=timepoints,
    idp_names=idp_names,
)
print("\nWITHIN SUBJECT VARIABILITY: BETWEEN TIMEPOINTS")
print(wsv)

# 3) Multivariate site differences using Mahalanobis distances
md = DiagnosticFunctions.MultiVariateBatchDifference_long(
    idp_matrix=idp_matrix,
    batch=batches,
    idp_names=idp_names,
)
print("\nMULTIVARIATE PAIRWISE SITE DIFFERENCES:")
print(md)

# 4) Mixed effects models -  batch effect (mean comparison); biological variability
mf = DiagnosticFunctions.MixedEffects_long(
    idp_matrix=idp_matrix,
    subjects=subjects,
    timepoints=timepoints,
    batches=batches,       
    idp_names=idp_names,
    covariates={"age": age},  # optional
    fix_eff=["age","timepoints"],   # batch is included automatically
    ran_eff=["subjects"],
    force_categorical=["timepoints"],
    force_numeric=["age"],
    zscore_var=["age"],
)
print("\nMIXED EFFECTS OUTPUTS:")
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
print(mf)  

# 5) Additive batch effects
addeff = DiagnosticFunctions.AdditiveEffect_long(
    idp_matrix=idp_matrix,
    subjects=subjects,
    timepoints=timepoints,
    batch_name=batches,
    idp_names=idp_names,
    covariates={"age": age},
    fix_eff=["age", "timepoints"],   # fixed effects
    ran_eff=["subjects"],            # random intercepts
    do_zscore=True,                  # z-score predictors AND response per feature
    reml=False,
    verbose=True,
)
print("\nRESULTS: ADDITIVE EFFECTS")
print(addeff)

# 6) Multiplicative batch effects
muleff = DiagnosticFunctions.MultiplicativeEffect_long(
    idp_matrix=idp_matrix,
    subjects=subjects,
    timepoints=timepoints,
    batch_name=batches,
    idp_names=idp_names,
    covariates={"age": age},
    fix_eff=["age", "timepoints"],   # fixed effects
    ran_eff=["subjects"],            # random intercepts
    do_zscore=True,                  # z-score predictors AND response per feature
    verbose=True,
)
print("\nRESULTS: MULTIPLICATIVE EFFECTS")
print(muleff) 
