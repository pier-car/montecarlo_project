"""
    WellMixed

Reference Julia implementation of the Thomson (1987) well-mixed Lagrangian
stochastic model for homogeneous Gaussian turbulence. This module is the
differential-testing oracle for the C++ compute core: it integrates the same
generalized Langevin equation with an independent implementation, so that the
optimised C++ port can be checked against it statistically.

Run the self-check with:

    julia julia/WellMixed.jl
"""
module WellMixed

using Random
using Statistics

export LangevinParams, simulate, well_mixed_residual

"""
    LangevinParams

Parameters for the homogeneous-turbulence Langevin integrator.

* `sigma2` : velocity variance σ² [m²/s²]
* `epsilon`: TKE dissipation rate ε [m²/s³]
* `c0`     : universal Lagrangian constant C₀
* `dt`     : integration step [s]
"""
struct LangevinParams
    sigma2::Float64
    epsilon::Float64
    c0::Float64
    dt::Float64
end

"""
    simulate(p, n_particles, n_steps; seed=1)

Integrate `n_particles` independent 1-D Ornstein–Uhlenbeck velocity processes
for `n_steps` steps. Returns the final velocity sample, whose variance must
remain close to `p.sigma2` (the well-mixed invariant).
"""
function simulate(p::LangevinParams, n_particles::Int, n_steps::Int; seed::Int=1)
    rng = MersenneTwister(seed)
    σ = sqrt(p.sigma2)
    u = σ .* randn(rng, n_particles)            # start well-mixed
    T_L = 2.0 * p.sigma2 / (p.c0 * p.epsilon)   # Lagrangian timescale
    b = sqrt(p.c0 * p.epsilon)
    sqrtdt = sqrt(p.dt)
    @inbounds for _ in 1:n_steps
        for i in eachindex(u)
            a = -u[i] / T_L                      # OU relaxation drift
            u[i] += a * p.dt + b * sqrtdt * randn(rng)
        end
    end
    return u
end

"""
    well_mixed_residual(p, n_particles, n_steps; seed=1)

Relative deviation of the simulated velocity variance from `p.sigma2`. For a
correct integrator this tends to zero as `n_particles → ∞`.
"""
function well_mixed_residual(p::LangevinParams, n_particles::Int, n_steps::Int;
                             seed::Int=1)
    u = simulate(p, n_particles, n_steps; seed=seed)
    return abs(var(u) - p.sigma2) / p.sigma2
end

end # module

# ---------------------------------------------------------------------------
# Self-check (executed when run as a script, not when imported).
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    using .WellMixed
    p = WellMixed.LangevinParams(0.5, 0.1, 4.0, 0.05)
    res = WellMixed.well_mixed_residual(p, 200_000, 200)
    println("well-mixed relative variance residual = ", res)
    @assert res < 0.05 "well-mixed invariant violated: residual=$res"
    println("WellMixed self-check passed.")
end
