import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from tqdm.auto import tqdm
from time import perf_counter
import functions_cpp

stack_tst = pd.read_feather("stack_tst_california.feather")

stack_trn = pd.read_feather("stack_trn_california.feather")

meta_learner = smf.ols(formula = "y ~ z1 + z2", data = stack_trn).fit()

y_hat_tst = meta_learner.predict(stack_tst[["z1", "z2"]])

Z = np.column_stack((np.ones(stack_trn.shape[0]), stack_trn[["z1", "z2"]].values))
Z_tst = np.column_stack((np.ones(stack_tst.shape[0]), stack_tst[["z1", "z2"]].values))

alpha = 0.1

print("Computing intervals...")

begin = perf_counter()

cp_interval2 = functions_cpp.full_cp_interval(Z, stack_trn["y"].values, Z_tst, alpha)

print(f"Elapsed: {perf_counter() - begin:.1f} seconds")

coverage2 = np.mean((cp_interval2[:, 0] <= stack_tst["y"].values) & (stack_tst["y"].values <= cp_interval2[:, 1]))
median_width2 = np.median(cp_interval2[:, 1] - cp_interval2[:, 0])

print(f"Coverage: {coverage2:.3f}")
print(f"Median width: {median_width2:.2f}")


