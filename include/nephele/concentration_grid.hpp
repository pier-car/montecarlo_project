#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace nephele {

/// Lightweight axis-aligned regular grid for accumulating particle mass into
/// concentration values. Kept header-only and dependency-free.
class ConcentrationGrid {
public:
    ConcentrationGrid(std::array<double, 3> origin,
                      std::array<double, 3> cell_size,
                      std::array<std::size_t, 3> dims)
        : origin_(origin), cell_size_(cell_size), dims_(dims) {
        for (int k = 0; k < 3; ++k) {
            if (cell_size_[k] <= 0.0) {
                throw std::invalid_argument("ConcentrationGrid: cell_size must be > 0");
            }
            if (dims_[k] == 0) {
                throw std::invalid_argument("ConcentrationGrid: dims must be > 0");
            }
        }
        cells_.assign(dims_[0] * dims_[1] * dims_[2], 0.0);
    }

    void reset() { std::fill(cells_.begin(), cells_.end(), 0.0); }

    /// Add `mass` to the cell containing (x, y, z). Returns false if the point
    /// lies outside the grid.
    bool deposit(double x, double y, double z, double mass) {
        const double fx = (x - origin_[0]) / cell_size_[0];
        const double fy = (y - origin_[1]) / cell_size_[1];
        const double fz = (z - origin_[2]) / cell_size_[2];
        if (fx < 0.0 || fy < 0.0 || fz < 0.0) {
            return false;
        }
        const auto ix = static_cast<std::size_t>(fx);
        const auto iy = static_cast<std::size_t>(fy);
        const auto iz = static_cast<std::size_t>(fz);
        if (ix >= dims_[0] || iy >= dims_[1] || iz >= dims_[2]) {
            return false;
        }
        cells_[index(ix, iy, iz)] += mass;
        return true;
    }

    /// Divide every accumulated cell by the cell volume to obtain concentration.
    void finalize_to_concentration() {
        const double volume = cell_size_[0] * cell_size_[1] * cell_size_[2];
        for (double& c : cells_) {
            c /= volume;
        }
    }

    [[nodiscard]] std::size_t index(std::size_t ix, std::size_t iy,
                                    std::size_t iz) const noexcept {
        return (iz * dims_[1] + iy) * dims_[0] + ix;
    }

    [[nodiscard]] const std::vector<double>& cells() const noexcept { return cells_; }
    [[nodiscard]] std::array<std::size_t, 3> dims() const noexcept { return dims_; }

private:
    std::array<double, 3> origin_;
    std::array<double, 3> cell_size_;
    std::array<std::size_t, 3> dims_;
    std::vector<double> cells_;
};

}  // namespace nephele
