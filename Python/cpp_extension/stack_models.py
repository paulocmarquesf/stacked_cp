import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

california = (
    pd.read_csv("california.csv")
      .assign(ocean_proximity = lambda df: df["ocean_proximity"].astype("category"))
      .rename(columns = {"median_house_value": "y"})
)

seed = 42
trn, tst = train_test_split(california, test_size = 0.30, random_state = seed)

alpha = 0.10

X_trn, y_trn = trn.drop(columns = "y"), trn["y"]
X_tst, y_tst = tst.drop(columns = "y"), tst["y"]

cat_cols = X_trn.select_dtypes(["category"]).columns.tolist()

# preparation of RF data by encoding categories
X_trn_rf = X_trn.copy()
X_tst_rf = X_tst.copy()
for col in cat_cols:
    X_trn_rf[col] = X_trn_rf[col].cat.codes
    X_tst_rf[col] = X_tst_rf[col].cat.codes

rf_full = RandomForestRegressor(n_estimators = 500, random_state = seed, n_jobs = -1)
rf_full.fit(X_trn_rf, y_trn)

trn_pool = Pool(data = X_trn, label = y_trn, cat_features = [X_trn.columns.get_loc(c) for c in cat_cols])
cb_full = CatBoostRegressor(random_seed = seed, verbose = 0)
cb_full.fit(trn_pool)

y_hat_tst_rf = rf_full.predict(X_tst_rf)
y_hat_tst_cb = cb_full.predict(Pool(data = X_tst, cat_features = [X_trn.columns.get_loc(c) for c in cat_cols]))

stack_tst = pd.DataFrame({"z1": y_hat_tst_cb, "z2": y_hat_tst_rf, "y": y_tst})

num_folds = 10
rng = np.random.default_rng(seed)
fold = rng.integers(0, num_folds, size = X_trn.shape[0])

print("Stacking the models...")

rows = []
for k in tqdm(range(num_folds), total = num_folds, desc = "fold", leave = False):
    out_of_sample = (fold == k)
    in_sample = ~out_of_sample
    X_in = X_trn.loc[in_sample]
    y_in = y_trn.loc[in_sample]
    X_out = X_trn.loc[out_of_sample]
    y_out = y_trn.loc[out_of_sample]

    # encode for RF
    X_in_rf = X_in.copy()
    X_out_rf = X_out.copy()
    for col in cat_cols:
        X_in_rf[col] = X_in_rf[col].cat.codes
        X_out_rf[col] = X_out_rf[col].cat.codes

    rf = RandomForestRegressor(n_estimators = 500, random_state = seed, n_jobs = -1)
    rf.fit(X_in_rf, y_in)
    cb = CatBoostRegressor(random_seed = seed, verbose = 0)
    cb.fit(Pool(data = X_in, label = y_in, cat_features = [X_trn.columns.get_loc(c) for c in cat_cols]))
    z1 = cb.predict(Pool(data = X_out, cat_features = [X_trn.columns.get_loc(c) for c in cat_cols]))
    z2 = rf.predict(X_out_rf)
    rows.append(pd.DataFrame({"z1": z1, "z2": z2, "y": y_out.values}))

stack_trn = pd.concat(rows, ignore_index = True)

stack_tst.to_feather("stack_tst_california.feather")

stack_trn.to_feather("stack_trn_california.feather")
