import numpy as np
from tqdm.auto import tqdm

def get_scores(z0, y0, Z, y, alpha, ZtZ_inv, beta_hat, ZtZ_new_inv):
    z0 = z0.reshape(-1, 1)

    beta_hat_new = beta_hat + ZtZ_new_inv @ z0 * (y0 - float(z0.T @ beta_hat))

    y_hat = Z @ beta_hat_new
    y0_hat = float(z0.T @ beta_hat_new)

    y_res = np.abs(y - y_hat)
    y0_res = abs(y0 - y0_hat)

    beta_res_hat = ZtZ_inv @ Z.T @ y_res
    beta_res_hat_new = beta_res_hat + ZtZ_new_inv @ z0 * (y0_res - float(z0.T @ beta_res_hat))

    delta_hat = Z @ beta_res_hat_new
    delta0_hat = float(z0.T @ beta_res_hat_new)

    r = y_res / (1 + delta_hat)

    n = Z.shape[0]
    index = int(np.ceil((1 - alpha) * (n + 1)) - 1)
    r_hat = np.sort(r, axis = 0)[index]

    r0 = y0_res / (1 + delta0_hat)

    return r0, r_hat

def full_cp_interval(Z, y, Z_tst, alpha, epsilon = 1e-2, sd_multiple = 5):
    y = y.reshape(-1, 1)
    sd_y = y.std()

    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    beta_hat = ZtZ_inv @ Z.T @ y

    cp_interval = np.zeros((Z_tst.shape[0], 2))

    for i in tqdm(range(Z_tst.shape[0]), desc = "row", leave = False):
        z0 = Z_tst[i, :].reshape(-1, 1)
        y0_guess = float(z0.T @ beta_hat)

        z0t_ZtZ_inv = z0.T @ ZtZ_inv
        ZtZ_new_inv = ZtZ_inv - (ZtZ_inv @ z0 @ z0t_ZtZ_inv) / (1 + float(z0t_ZtZ_inv @ z0))

        inf = y0_guess
        lower = inf - sd_multiple * sd_y

        while inf - lower > epsilon:
            y0 = float((lower + inf) / 2)
            r0, r_hat = get_scores(z0, y0, Z, y, alpha, ZtZ_inv, beta_hat, ZtZ_new_inv)

            if r0 <= r_hat:
                inf = y0
            else:
                lower = y0

        cp_interval[i, 0] = inf

        sup = y0_guess
        upper = sup + sd_multiple * sd_y

        while upper - sup > epsilon:
            y0 = float((sup + upper) / 2)
            r0, r_hat = get_scores(z0, y0, Z, y, alpha, ZtZ_inv, beta_hat, ZtZ_new_inv)

            if r0 <= r_hat:
                sup = y0
            else:
                upper = y0

        cp_interval[i, 1] = sup

    return cp_interval
