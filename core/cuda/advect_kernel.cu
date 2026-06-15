// Optional CUDA advection kernel for the Lagrangian engine.
//
// This kernel is compiled only when NEPHELE_USE_CUDA=ON. It mirrors the
// per-particle Langevin update in core/src/lagrangian_engine.cpp for a uniform
// turbulence field, which is the common case for the GPU-accelerated path.
//
// The host-side glue that launches this kernel lives behind the same CMake
// option; when CUDA is disabled the CPU OpenMP path is used transparently.

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace nephele {
namespace cuda {

extern "C" __global__ void advect_uniform_kernel(
    double* __restrict__ x, double* __restrict__ y, double* __restrict__ z,
    double* __restrict__ ux, double* __restrict__ uy, double* __restrict__ uz,
    double* __restrict__ mass, unsigned long long n, double dt, double sqrt_dt,
    double c0, double epsilon, double mass_decay, double ground,
    double mean_wx, double mean_wy, double mean_wz, double sig2x, double sig2y,
    double sig2z, unsigned long long seed) {
    const unsigned long long i =
        blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    if (i >= n) {
        return;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(seed, i, 0, &state);

    const double b = sqrt(c0 * epsilon);
    const double sig2[3] = {sig2x, sig2y, sig2z};
    double u[3] = {ux[i], uy[i], uz[i]};

#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const double t_l = 2.0 * sig2[k] / (c0 * epsilon);
        const double a = -u[k] / t_l;  // uniform field: no inhomogeneity drift
        const double xi = curand_normal_double(&state);
        u[k] += a * dt + b * sqrt_dt * xi;
    }
    ux[i] = u[0];
    uy[i] = u[1];
    uz[i] = u[2];

    x[i] += (mean_wx + u[0]) * dt;
    y[i] += (mean_wy + u[1]) * dt;
    z[i] += (mean_wz + u[2]) * dt;

    if (z[i] < ground) {
        z[i] = ground + (ground - z[i]);
        uz[i] = -uz[i];
    }

    mass[i] *= mass_decay;
}

}  // namespace cuda
}  // namespace nephele
