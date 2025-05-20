#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <atomic>
#include <iomanip>

namespace py = pybind11;
using arma::mat;
using arma::colvec;
using arma::rowvec;
using arma::vec;

static vec get_scores(const colvec& z0,
                      double y0,
                      const mat& Z,
                      const colvec& y,
                      double alpha,
                      const mat& ZtZ_inv,
                      const colvec& beta_hat,
                      const mat& ZtZ_new_inv)
{
    colvec beta_hat_new = beta_hat + ZtZ_new_inv * z0 * (y0 - arma::dot(z0, beta_hat));
    colvec y_hat = Z * beta_hat_new;
    double y0_hat = arma::dot(z0, beta_hat_new);
    colvec y_res = arma::abs(y - y_hat);
    double y0_res = std::abs(y0 - y0_hat);
    colvec beta_res_hat = ZtZ_inv * (Z.t() * y_res);
    colvec beta_res_hat_new = beta_res_hat + ZtZ_new_inv * z0 * (y0_res - arma::dot(z0, beta_res_hat));
    colvec delta_hat = Z * beta_res_hat_new;
    double delta0_hat = arma::dot(z0, beta_res_hat_new);
    vec r(y_res.n_elem);
    for (arma::uword i = 0; i < y_res.n_elem; ++i) r(i) = y_res(i) / (1.0 + delta_hat(i));
    double r0 = y0_res / (1.0 + delta0_hat);
    arma::uword n = r.n_elem;
    arma::uword idx = static_cast<arma::uword>(std::ceil((1.0 - alpha) * (n + 1)) - 1);
    if (idx >= n) idx = n - 1;
    vec r_sorted = arma::sort(r);
    double r_hat = r_sorted(idx);
    vec out(2);
    out(0) = r0;
    out(1) = r_hat;
    return out;
}

py::array_t<double> full_cp_interval(
    py::array_t<double, py::array::c_style | py::array::forcecast> Z_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> y_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> Ztst_in,
    double alpha,
    double epsilon = 1e-2,
    double sd_multiple = 5.0)
{
    int n = Z_in.shape(0);
    int p = Z_in.shape(1);
    int m = Ztst_in.shape(0);
    mat Z_tmp(Z_in.mutable_data(), p, n, false, false);
    mat Z = Z_tmp.t();
    colvec y(y_in.mutable_data(), n);
    mat Ztst_tmp(Ztst_in.mutable_data(), p, m, false, false);
    mat Ztst = Ztst_tmp.t();
    double sd_y = arma::stddev(y, 0);
    mat ZtZ_inv = arma::inv_sympd(Z.t() * Z);
    colvec beta_hat = ZtZ_inv * (Z.t() * y);
    py::array_t<double> out({m, 2});
    auto buf = out.mutable_unchecked<2>();
    std::atomic<int> counter(0);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < m; ++i) {
        colvec z0 = Ztst.row(i).t();
        double y0_guess = arma::dot(z0, beta_hat);
        rowvec z0tZtZ = z0.t() * ZtZ_inv;
        double denom = 1.0 + arma::as_scalar(z0tZtZ * z0);
        mat ZtZ_new_inv = ZtZ_inv - (ZtZ_inv * z0 * z0tZtZ) / denom;
        double lower = y0_guess - sd_multiple * sd_y;
        double inf = y0_guess;
        while (inf - lower > epsilon) {
            double y0 = 0.5 * (lower + inf);
            vec s = get_scores(z0, y0, Z, y, alpha, ZtZ_inv, beta_hat, ZtZ_new_inv);
            if (s(0) <= s(1)) inf = y0; else lower = y0;
        }
        double upper = y0_guess + sd_multiple * sd_y;
        double sup = y0_guess;
        while (upper - sup > epsilon) {
            double y0 = 0.5 * (sup + upper);
            vec s = get_scores(z0, y0, Z, y, alpha, ZtZ_inv, beta_hat, ZtZ_new_inv);
            if (s(0) <= s(1)) sup = y0; else upper = y0;
        }
        buf(i, 0) = inf;
        buf(i, 1) = sup;
        int c = ++counter;
        if (c % 1000 == 0 || c == m) {
            #pragma omp critical
            {
                double pct = 100.0 * c / m;
                std::cerr << "\rProgress: " << std::fixed << std::setprecision(1) << pct << "%";
                if (c == m) std::cerr << std::endl;
            }
        }
    }
    return out;
}

PYBIND11_MODULE(functions_cpp, m) {
    m.def("full_cp_interval", &full_cp_interval,
          py::arg("Z"), py::arg("y"), py::arg("Ztst"), py::arg("alpha"),
          py::arg("epsilon") = 1e-2, py::arg("sd_multiple") = 5.0);
}
