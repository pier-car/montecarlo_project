#include "nephele/ensemble_kalman.hpp"

#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace nephele {

namespace {

// Solve the symmetric positive-definite system A x = b in place using a
// Cholesky decomposition. `a` is m*m row-major and is overwritten. Throws if A
// is not positive definite. Returns the solution vector.
std::vector<double> spd_solve(std::vector<double> a, std::vector<double> b,
                              std::size_t m) {
    // Cholesky: A = L L^T, lower triangular stored in `a`.
    for (std::size_t j = 0; j < m; ++j) {
        double diag = a[j * m + j];
        for (std::size_t k = 0; k < j; ++k) {
            diag -= a[j * m + k] * a[j * m + k];
        }
        if (diag <= 0.0) {
            throw std::runtime_error(
                "EnsembleKalmanFilter: innovation covariance not positive definite");
        }
        const double ljj = std::sqrt(diag);
        a[j * m + j] = ljj;
        for (std::size_t i = j + 1; i < m; ++i) {
            double s = a[i * m + j];
            for (std::size_t k = 0; k < j; ++k) {
                s -= a[i * m + k] * a[j * m + k];
            }
            a[i * m + j] = s / ljj;
        }
    }

    // Forward substitution: L y = b.
    std::vector<double> y(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        double s = b[i];
        for (std::size_t k = 0; k < i; ++k) {
            s -= a[i * m + k] * y[k];
        }
        y[i] = s / a[i * m + i];
    }

    // Back substitution: L^T x = y.
    std::vector<double> x(m, 0.0);
    for (std::size_t ii = 0; ii < m; ++ii) {
        const std::size_t i = m - 1 - ii;
        double s = y[i];
        for (std::size_t k = i + 1; k < m; ++k) {
            s -= a[k * m + i] * x[k];
        }
        x[i] = s / a[i * m + i];
    }
    return x;
}

}  // namespace

EnsembleKalmanFilter::EnsembleKalmanFilter(std::size_t state_dim,
                                           std::size_t ensemble_size)
    : state_dim_(state_dim), ensemble_size_(ensemble_size) {
    if (state_dim_ == 0) {
        throw std::invalid_argument("EnsembleKalmanFilter: state_dim must be > 0");
    }
    if (ensemble_size_ < 2) {
        throw std::invalid_argument("EnsembleKalmanFilter: ensemble_size must be >= 2");
    }
    ensemble_.assign(ensemble_size_, std::vector<double>(state_dim_, 0.0));
}

void EnsembleKalmanFilter::set_ensemble(
    const std::vector<std::vector<double>>& members) {
    if (members.size() != ensemble_size_) {
        throw std::invalid_argument("EnsembleKalmanFilter: wrong ensemble size");
    }
    for (const auto& m : members) {
        if (m.size() != state_dim_) {
            throw std::invalid_argument("EnsembleKalmanFilter: wrong state dimension");
        }
    }
    ensemble_ = members;
}

std::vector<double> EnsembleKalmanFilter::mean() const {
    std::vector<double> mu(state_dim_, 0.0);
    for (const auto& member : ensemble_) {
        for (std::size_t j = 0; j < state_dim_; ++j) {
            mu[j] += member[j];
        }
    }
    const double inv = 1.0 / static_cast<double>(ensemble_size_);
    for (double& v : mu) {
        v *= inv;
    }
    return mu;
}

void EnsembleKalmanFilter::update(const std::vector<double>& obs_operator,
                                  const std::vector<double>& observations,
                                  const std::vector<double>& obs_var,
                                  std::uint64_t seed) {
    const std::size_t m = observations.size();
    if (m == 0) {
        throw std::invalid_argument("EnsembleKalmanFilter: no observations");
    }
    if (obs_operator.size() != m * state_dim_) {
        throw std::invalid_argument("EnsembleKalmanFilter: H has wrong shape");
    }
    if (obs_var.size() != m) {
        throw std::invalid_argument("EnsembleKalmanFilter: obs_var has wrong length");
    }
    for (double r : obs_var) {
        if (r <= 0.0) {
            throw std::invalid_argument("EnsembleKalmanFilter: obs_var must be > 0");
        }
    }

    const std::size_t ne = ensemble_size_;
    const std::size_t n = state_dim_;
    const double inv_ne_1 = 1.0 / static_cast<double>(ne - 1);

    // Forecast mean.
    const std::vector<double> xbar = mean();

    // State anomalies A (ne x n) and predicted-observation anomalies HA (ne x m).
    std::vector<double> hx(ne * m, 0.0);     // H * x for each member
    std::vector<double> hxbar(m, 0.0);       // H * xbar
    for (std::size_t i = 0; i < m; ++i) {
        double acc = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            acc += obs_operator[i * n + j] * xbar[j];
        }
        hxbar[i] = acc;
    }
    for (std::size_t e = 0; e < ne; ++e) {
        for (std::size_t i = 0; i < m; ++i) {
            double acc = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                acc += obs_operator[i * n + j] * ensemble_[e][j];
            }
            hx[e * m + i] = acc;
        }
    }

    // Innovation covariance S = HA HA^T / (ne-1) + R   (m x m).
    std::vector<double> s(m * m, 0.0);
    for (std::size_t a = 0; a < m; ++a) {
        for (std::size_t b = 0; b < m; ++b) {
            double acc = 0.0;
            for (std::size_t e = 0; e < ne; ++e) {
                acc += (hx[e * m + a] - hxbar[a]) * (hx[e * m + b] - hxbar[b]);
            }
            s[a * m + b] = acc * inv_ne_1;
        }
        s[a * m + a] += obs_var[a];
    }

    // Cross covariance C = A^T HA / (ne-1)  (n x m).
    std::vector<double> cross(n * m, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            double acc = 0.0;
            for (std::size_t e = 0; e < ne; ++e) {
                acc += (ensemble_[e][j] - xbar[j]) * (hx[e * m + i] - hxbar[i]);
            }
            cross[j * m + i] = acc * inv_ne_1;
        }
    }

    // Perturbed observations and analysis update per member.
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);

    for (std::size_t e = 0; e < ne; ++e) {
        // d = y + eta - H x_e
        std::vector<double> d(m, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            const double eta = std::sqrt(obs_var[i]) * gauss(rng);
            d[i] = observations[i] + eta - hx[e * m + i];
        }

        // Solve S w = d.
        const std::vector<double> w = spd_solve(s, d, m);

        // x_e += C w.
        for (std::size_t j = 0; j < n; ++j) {
            double delta = 0.0;
            for (std::size_t i = 0; i < m; ++i) {
                delta += cross[j * m + i] * w[i];
            }
            ensemble_[e][j] += delta;
        }
    }
}

}  // namespace nephele
