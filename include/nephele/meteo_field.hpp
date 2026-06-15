#pragma once

#include <array>
#include <stdexcept>

namespace nephele {

/// Local turbulence/meteorology state sampled at a point.
struct MeteoState {
    std::array<double, 3> mean_wind;   ///< U_i  [m/s]
    std::array<double, 3> sigma2;      ///< sigma_i^2 velocity variance [m^2/s^2]
    std::array<double, 3> dsigma2_dx;  ///< d(sigma_i^2)/dx_i  [m/s^2]
    double epsilon;                    ///< TKE dissipation rate [m^2/s^3]
};

/// Abstract meteorological field. Implementations are provided by the Python
/// pre-processor (gridded interpolation of QGIS-derived canopy + met data).
class MeteoField {
public:
    virtual ~MeteoField() = default;

    /// Sample the meteorological state at world position (x, y, z).
    /// Implementations must return strictly positive sigma2 components and a
    /// strictly positive epsilon.
    virtual MeteoState sample(double x, double y, double z) const = 0;

    /// Ground elevation at (x, y) used for the reflecting lower boundary.
    virtual double ground_height(double x, double y) const = 0;
};

/// Homogeneous, stationary turbulence -- reference field for verification and
/// for regions outside the resolved canopy.
class UniformMeteoField final : public MeteoField {
public:
    UniformMeteoField(std::array<double, 3> mean_wind,
                      std::array<double, 3> sigma2,
                      double epsilon,
                      double ground = 0.0)
        : mean_wind_(mean_wind),
          sigma2_(sigma2),
          epsilon_(epsilon),
          ground_(ground) {
        if (epsilon_ <= 0.0) {
            throw std::invalid_argument("UniformMeteoField: epsilon must be > 0");
        }
        for (double s : sigma2_) {
            if (s <= 0.0) {
                throw std::invalid_argument(
                    "UniformMeteoField: sigma2 components must be > 0");
            }
        }
    }

    MeteoState sample(double, double, double) const override {
        return MeteoState{mean_wind_, sigma2_, {0.0, 0.0, 0.0}, epsilon_};
    }

    double ground_height(double, double) const override { return ground_; }

private:
    std::array<double, 3> mean_wind_;
    std::array<double, 3> sigma2_;
    double epsilon_;
    double ground_;
};

}  // namespace nephele
