#include "nephele/lagrangian_engine.hpp"

#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nephele {

namespace {

inline int max_threads() {
#ifdef _OPENMP
    return std::max(1, omp_get_max_threads());
#else
    return 1;
#endif
}

inline int this_thread() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

}  // namespace

void LagrangianDispersionEngine::allocate_and_seed() {
    const std::size_t n = cfg_.num_particles;

    // Single up-front allocation for every SoA buffer.
    x_.assign(n, cfg_.source_pos[0]);
    y_.assign(n, cfg_.source_pos[1]);
    z_.assign(n, cfg_.source_pos[2]);
    mass_.assign(n, cfg_.source_mass / static_cast<double>(n));

    ux_.assign(n, 0.0);
    uy_.assign(n, 0.0);
    uz_.assign(n, 0.0);

    // Initialise turbulent velocities from the local ambient PDF so the ensemble
    // starts well-mixed.
    const MeteoState m0 =
        meteo_->sample(cfg_.source_pos[0], cfg_.source_pos[1], cfg_.source_pos[2]);

    std::seed_seq seq{static_cast<std::uint32_t>(cfg_.seed),
                      static_cast<std::uint32_t>(cfg_.seed >> 32)};
    std::mt19937_64 init_rng(seq);
    std::normal_distribution<double> g(0.0, 1.0);
    for (std::size_t i = 0; i < n; ++i) {
        ux_[i] = std::sqrt(m0.sigma2[0]) * g(init_rng);
        uy_[i] = std::sqrt(m0.sigma2[1]) * g(init_rng);
        uz_[i] = std::sqrt(m0.sigma2[2]) * g(init_rng);
    }

    // Deterministically derive an independent RNG per thread from the master seed.
    const int nthreads = max_threads();
    rngs_.clear();
    rngs_.reserve(static_cast<std::size_t>(nthreads));
    for (int t = 0; t < nthreads; ++t) {
        std::seed_seq tseq{static_cast<std::uint32_t>(cfg_.seed),
                           static_cast<std::uint32_t>(cfg_.seed >> 32),
                           static_cast<std::uint32_t>(t)};
        rngs_.emplace_back(tseq);
    }
}

void LagrangianDispersionEngine::step() {
    const std::size_t n = cfg_.num_particles;
    const double dt = cfg_.dt;
    const double c0 = cfg_.c0;
    const double sqrt_dt = std::sqrt(dt);
    const double mass_decay = std::exp(-cfg_.decay_rate * dt);

    // SoA base pointers hoisted out of the loop.
    double* __restrict px = x_.data();
    double* __restrict py = y_.data();
    double* __restrict pz = z_.data();
    double* __restrict pux = ux_.data();
    double* __restrict puy = uy_.data();
    double* __restrict puz = uz_.data();
    double* __restrict pm = mass_.data();

#pragma omp parallel
    {
        std::mt19937_64& rng = rngs_[static_cast<std::size_t>(this_thread())];
        std::normal_distribution<double> gauss(0.0, 1.0);

#pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            const MeteoState m = meteo_->sample(px[i], py[i], pz[i]);

            // Diffusion amplitude b = sqrt(C0 * epsilon).
            const double b = std::sqrt(c0 * m.epsilon);

            // Per-component update of the generalized Langevin equation.
            double u[3] = {pux[i], puy[i], puz[i]};
            for (int k = 0; k < 3; ++k) {
                const double sigma2 = m.sigma2[k];
                // Lagrangian timescale T_L = 2 sigma^2 / (C0 epsilon).
                const double t_l = 2.0 * sigma2 / (c0 * m.epsilon);

                // Thomson well-mixed drift: OU relaxation + inhomogeneity drift.
                const double relax = -u[k] / t_l;
                const double inhom =
                    0.5 * (1.0 + (u[k] * u[k]) / sigma2) * m.dsigma2_dx[k];
                const double a = relax + inhom;

                u[k] += a * dt + b * sqrt_dt * gauss(rng);
            }
            pux[i] = u[0];
            puy[i] = u[1];
            puz[i] = u[2];

            // Advect position with mean wind + turbulent fluctuation.
            px[i] += (m.mean_wind[0] + u[0]) * dt;
            py[i] += (m.mean_wind[1] + u[1]) * dt;
            pz[i] += (m.mean_wind[2] + u[2]) * dt;

            // Reflecting lower boundary at the ground surface.
            const double zg = meteo_->ground_height(px[i], py[i]);
            if (pz[i] < zg) {
                pz[i] = zg + (zg - pz[i]);  // mirror position
                puz[i] = -puz[i];           // mirror vertical velocity
            }

            // First-order agent decay / deposition.
            pm[i] *= mass_decay;
        }
    }

    time_ += dt;
}

std::vector<double> LagrangianDispersionEngine::concentration(
    std::array<double, 3> origin,
    std::array<double, 3> cell_size,
    std::array<std::size_t, 3> dims) const {

    for (int k = 0; k < 3; ++k) {
        if (cell_size[k] <= 0.0) {
            throw std::invalid_argument("concentration: cell_size must be > 0");
        }
        if (dims[k] == 0) {
            throw std::invalid_argument("concentration: dims must be > 0");
        }
    }

    const std::size_t ncells = dims[0] * dims[1] * dims[2];
    std::vector<double> grid(ncells, 0.0);
    const double cell_volume = cell_size[0] * cell_size[1] * cell_size[2];

    const std::size_t n = cfg_.num_particles;
    for (std::size_t i = 0; i < n; ++i) {
        const double fx = (x_[i] - origin[0]) / cell_size[0];
        const double fy = (y_[i] - origin[1]) / cell_size[1];
        const double fz = (z_[i] - origin[2]) / cell_size[2];

        if (fx < 0.0 || fy < 0.0 || fz < 0.0) {
            continue;
        }
        const auto ix = static_cast<std::size_t>(fx);
        const auto iy = static_cast<std::size_t>(fy);
        const auto iz = static_cast<std::size_t>(fz);
        if (ix >= dims[0] || iy >= dims[1] || iz >= dims[2]) {
            continue;
        }

        const std::size_t idx = (iz * dims[1] + iy) * dims[0] + ix;
        grid[idx] += mass_[i] / cell_volume;
    }

    return grid;
}

}  // namespace nephele
