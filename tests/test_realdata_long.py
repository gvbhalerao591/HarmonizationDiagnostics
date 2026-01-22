import numpy as np
import pandas as pd

from DiagnoseHarmonization import DiagnosticFunctions

# Load CSV
df = pd.read_csv("/Users/psyc1586_admin/HarmonisationTool/jakeRepo2/onharmony.csv")

# ---- REQUIRED STRUCTURAL VARIABLES ----
subjects   = df["subject"].astype(str).tolist()
timepoints = df["timepoint"].astype(str).tolist()
batches    = df["scan_session"].astype(str).tolist()
age = df["age"].astype(float).tolist()
sex = df["sex"].astype(str).tolist()

# ---- COVARIATES (dict) ----
covariates = {
    "age": age,
    "sex": sex
}

# ---- IDPs ----
idp_names = ["T1_SIENAX_peripheral_GM_norm_vol"]   # or infer automatically (see below)
idp_matrix = df[idp_names].to_numpy(dtype=float)

# ---- SANITY CHECKS ----
n_samples = len(df)
print("n_samples:", n_samples)
print("shape idp_matrix:", idp_matrix.shape)
print("subjects (unique):", sorted(set(subjects)))
print("batches (unique):", sorted(set(batches)))


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
    covariates=covariates,  # optional
    fix_eff=["age","sex"],   # batch is included automatically
    ran_eff=["subjects"],
    force_categorical=["sex"],
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
    covariates=covariates,
    fix_eff=["age", "sex"],   # fixed effects
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
    covariates=covariates,
    fix_eff=["age", "sex"],   # fixed effects
    ran_eff=["subjects"],            # random intercepts
    do_zscore=True,                  # z-score predictors AND response per feature
    verbose=True,
)
print("\nRESULTS: MULTIPLICATIVE EFFECTS")
print(muleff)  
