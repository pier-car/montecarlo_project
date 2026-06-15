#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nephele {

/// Stochastic Ensemble Kalman Filter (perturbed-observations variant).
///
/// Maintains an ensemble of state vectors and assimilates linear observations
/// y = H x + noise. The forecast covariance is estimated directly from the
/// ensemble, so no tangent-linear model is required.
class EnsembleKalmanFilter {
public:
    /// @param state_dim   dimension n of each state vector
    /// @param ensemble_size  number of ensemble members N_e (>= 2)
    EnsembleKalmanFilter(std::size_t state_dim, std::size_t ensemble_size);

    /// Replace the current ensemble. `members` must have `ensemble_size` rows,
    /// each of length `state_dim` (row-major).
    void set_ensemble(const std::vector<std::vector<double>>& members);

    /// Analysis update.
    ///
    /// @param obs_operator  observation matrix H, shape (m x n) row-major
    /// @param observations  measured vector y, length m
    /// @param obs_var       diagonal observation-error variances R, length m
    /// @param seed          RNG seed for observation perturbations
    void update(const std::vector<double>& obs_operator,
                const std::vector<double>& observations,
                const std::vector<double>& obs_var,
                std::uint64_t seed = 0xA11CEULL);

    /// Ensemble mean (length state_dim).
    [[nodiscard]] std::vector<double> mean() const;

    [[nodiscard]] const std::vector<std::vector<double>>& ensemble() const noexcept {
        return ensemble_;
    }

    [[nodiscard]] std::size_t state_dim() const noexcept { return state_dim_; }
    [[nodiscard]] std::size_t ensemble_size() const noexcept { return ensemble_size_; }

private:
    std::size_t state_dim_;
    std::size_t ensemble_size_;
    std::vector<std::vector<double>> ensemble_;  // N_e rows of length n
};

}  // namespace nephele
