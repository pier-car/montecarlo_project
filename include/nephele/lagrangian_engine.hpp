#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "nephele/meteo_field.hpp"

namespace nephele {

/// Configuration for the Lagrangian stochastic dispersion engine.
struct EngineConfig {
    std::size_t num_particles = 100'000;            ///< ensemble size N_p
    double dt = 0.1;                                ///< integration step [s]
    double c0 = 4.0;                                ///< universal Lagrangian constant
    double decay_rate = 0.0;                        ///< agent decay/deposition lambda [1/s]
    std::array<double, 3> source_pos{0.0, 0.0, 1.0};///< release location [m]
    double source_mass = 1.0;                       ///< total released mass/activity q
    std::uint64_t seed = 0xC0FFEEULL;               ///< master RNG seed (reproducible)

    void validate() const {
        if (num_particles == 0) {
            throw std::invalid_argument("EngineConfig: num_particles must be > 0");
        }
        if (dt <= 0.0) {
            throw std::invalid_argument("EngineConfig: dt must be > 0");
        }
        if (c0 <= 0.0) {
            throw std::invalid_argument("EngineConfig: c0 must be > 0");
        }
        if (decay_rate < 0.0) {
            throw std::invalid_argument("EngineConfig: decay_rate must be >= 0");
        }
    }
};

/// Faster-than-real-time Lagrangian stochastic dispersion engine.
///
/// Integrates the Thomson (1987) well-mixed generalized Langevin equation for an
/// ensemble of marker particles. Memory is laid out Structure-of-Arrays for
/// cache efficiency; no heap allocation occurs inside the time loop.
class LagrangianDispersionEngine {
public:
    LagrangianDispersionEngine(EngineConfig config,
                               std::shared_ptr<const MeteoField> meteo)
        : cfg_(config), meteo_(std::move(meteo)), time_(0.0) {
        cfg_.validate();
        if (!meteo_) {
            throw std::invalid_argument(
                "LagrangianDispersionEngine: meteo field must not be null");
        }
        allocate_and_seed();
    }

    /// Advance the entire ensemble by one time step dt.
    void step();

    /// Advance the ensemble by `n` steps.
    void advance(std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            step();
        }
    }

    /// Bin particle mass into a regular grid, returning concentration [mass/m^3].
    ///
    /// The grid is axis-aligned with `dims[0]*dims[1]*dims[2]` cells of size
    /// (dx,dy,dz) anchored at `origin`. Particles outside the grid are ignored.
    std::vector<double> concentration(std::array<double, 3> origin,
                                      std::array<double, 3> cell_size,
                                      std::array<std::size_t, 3> dims) const;

    [[nodiscard]] double time() const noexcept { return time_; }
    [[nodiscard]] std::size_t size() const noexcept { return cfg_.num_particles; }

    // Read-only SoA accessors (zero-copy exposure to Python bindings).
    [[nodiscard]] const std::vector<double>& x() const noexcept { return x_; }
    [[nodiscard]] const std::vector<double>& y() const noexcept { return y_; }
    [[nodiscard]] const std::vector<double>& z() const noexcept { return z_; }
    [[nodiscard]] const std::vector<double>& mass() const noexcept { return mass_; }

private:
    void allocate_and_seed();

    EngineConfig cfg_;
    std::shared_ptr<const MeteoField> meteo_;
    double time_;

    // Structure-of-Arrays particle state (contiguous, cache-friendly).
    std::vector<double> x_, y_, z_;     // positions
    std::vector<double> ux_, uy_, uz_;  // turbulent velocity fluctuations
    std::vector<double> mass_;          // per-particle remaining mass

    // One RNG per OpenMP thread, allocated once (no per-step allocation).
    std::vector<std::mt19937_64> rngs_;
};

}  // namespace nephele
