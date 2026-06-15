#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "nephele/ensemble_kalman.hpp"

using namespace nephele;

// A scalar observation that directly measures the (single) state component must
// pull the ensemble mean toward the observation.
TEST(EnKF, ScalarUpdateMovesMeanTowardObservation) {
    EnsembleKalmanFilter enkf(1, 4000);

    std::vector<std::vector<double>> members;
    members.reserve(4000);
    // Forecast ensemble centred at 0 with spread 1.
    double phase = 0.0;
    for (std::size_t e = 0; e < 4000; ++e) {
        phase += 0.6180339887;  // low-discrepancy-ish deterministic spread
        const double g = std::sqrt(-2.0 * std::log((e + 1.0) / 4001.0)) *
                         std::cos(6.2831853 * std::fmod(phase, 1.0));
        members.push_back({g});
    }
    enkf.set_ensemble(members);

    const double prior_mean = enkf.mean()[0];

    // Observe y = 5 with H = [1], R = 0.25.
    enkf.update({1.0}, {5.0}, {0.25}, 42);

    const double post_mean = enkf.mean()[0];
    EXPECT_GT(post_mean, prior_mean);
    EXPECT_LT(post_mean, 5.0);  // shrinks toward but not past the observation
    EXPECT_GT(post_mean, 2.0);  // meaningful correction given R < prior var
}

TEST(EnKF, RejectsInvalidShapes) {
    EnsembleKalmanFilter enkf(2, 10);
    EXPECT_THROW(enkf.update({1.0}, {1.0, 2.0}, {0.1, 0.1}), std::invalid_argument);
    EXPECT_THROW(enkf.update({1.0, 0.0}, {1.0}, {-0.1}), std::invalid_argument);
}

TEST(EnKF, ConstructorValidatesDimensions) {
    EXPECT_THROW(EnsembleKalmanFilter(0, 10), std::invalid_argument);
    EXPECT_THROW(EnsembleKalmanFilter(2, 1), std::invalid_argument);
}
