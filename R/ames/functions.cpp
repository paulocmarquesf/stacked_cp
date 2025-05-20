#include <RcppArmadillo.h>
#include <omp.h>
#include <iomanip>           // ⟵ new
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

static vec get_scores(const colvec& z0, double y0, const mat& Z, const colvec& y,
                      double alpha, const mat& ZtZ_inv,
                      const colvec& beta_hat, const mat& ZtZ_new_inv) {
    colvec beta_hat_new = beta_hat + ZtZ_new_inv * z0 * (y0 - dot(z0, beta_hat));
    colvec y_hat = Z * beta_hat_new;
    double y0_hat = dot(z0, beta_hat_new);

    colvec y_res = abs(y - y_hat);
    double y0_res = std::abs(y0 - y0_hat);

    colvec beta_res_hat = ZtZ_inv * (Z.t() * y_res);
    colvec beta_res_hat_new = beta_res_hat + ZtZ_new_inv * z0 * (y0_res - dot(z0, beta_res_hat));

    colvec delta_hat = Z * beta_res_hat_new;
    double delta0_hat = dot(z0, beta_res_hat_new);

    colvec r = y_res / (1.0 + delta_hat);
    double r0 = y0_res / (1.0 + delta0_hat);

    size_t n = r.n_elem;
    size_t k = std::ceil((1.0 - alpha) * (n + 1.0)) - 1;
    if (k >= n) k = n - 1;
    uvec ord = sort_index(r);
    double r_hat = r(ord(k));

    vec out(2);
    out[0] = r0;
    out[1] = r_hat;
    return out;
}

// [[Rcpp::export]]
mat full_cp_interval_Rcpp(const mat& Z, const colvec& y, const mat& Z_tst, double alpha,
                          double epsilon = 1e-2, double sd_multiple = 5.0) {
    size_t m = Z_tst.n_rows;
    double sd_y = stddev(y);

    mat ZtZ_inv = inv_sympd(Z.t() * Z);
    colvec beta_hat = ZtZ_inv * (Z.t() * y);

    mat cp_interval(m, 2, fill::zeros);

    size_t counter = 0;                              // ⟵ new
    size_t step = std::max<size_t>(1, m / 100);      // ⟵ new

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < m; ++i) {
        colvec z0 = Z_tst.row(i).t();
        double y0_guess = dot(z0, beta_hat);

        rowvec z0t_ZtZ_inv = z0.t() * ZtZ_inv;
        mat ZtZ_new_inv = ZtZ_inv - (ZtZ_inv * z0 * z0t_ZtZ_inv) / (1.0 + as_scalar(z0t_ZtZ_inv * z0));

        double inf = y0_guess;
        double lower = inf - sd_multiple * sd_y;
        while ((inf - lower) > epsilon) {
            double y0 = 0.5 * (lower + inf);
            vec s = get_scores(z0, y0, Z, y, alpha, ZtZ_inv, beta_hat, ZtZ_new_inv);
            if (s[0] <= s[1]) inf = y0;
            else lower = y0;
        }
        cp_interval(i, 0) = inf;

        double sup = y0_guess;
        double upper = sup + sd_multiple * sd_y;
        while ((upper - sup) > epsilon) {
            double y0 = 0.5 * (sup + upper);
            vec s = get_scores(z0, y0, Z, y, alpha, ZtZ_inv, beta_hat, ZtZ_new_inv);
            if (s[0] <= s[1]) sup = y0;
            else upper = y0;
        }
        cp_interval(i, 1) = sup;

        #pragma omp atomic                             // ⟵ new
        ++counter;                                     // ⟵ new
        if (counter % step == 0 || counter == m) {     // ⟵ new
            #pragma omp critical                       // ⟵ new
            {
                Rcout << "\r" << std::setw(3)
                      << (counter * 100) / m << "% completed" << std::flush;
                if (counter == m) Rcout << std::endl;
            }
        }
    }
    return cp_interval;
}
