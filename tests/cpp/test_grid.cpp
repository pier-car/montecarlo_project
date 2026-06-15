#include <gtest/gtest.h>

#include <memory>
#include <numeric>

#include "nephele/concentration_grid.hpp"
#include "nephele/lagrangian_engine.hpp"
#include "nephele/meteo_field.hpp"

using namespace nephele;

TEST(ConcentrationGrid, ConservesDepositedMass) {
    ConcentrationGrid grid({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {4, 4, 4});
    EXPECT_TRUE(grid.deposit(0.5, 0.5, 0.5, 2.0));
    EXPECT_TRUE(grid.deposit(3.5, 3.5, 3.5, 3.0));
    EXPECT_FALSE(grid.deposit(-1.0, 0.0, 0.0, 1.0));  // outside
    EXPECT_FALSE(grid.deposit(10.0, 0.0, 0.0, 1.0));  // outside

    const double total =
        std::accumulate(grid.cells().begin(), grid.cells().end(), 0.0);
    EXPECT_DOUBLE_EQ(total, 5.0);  // before volume normalisation, cell vol == 1

    grid.finalize_to_concentration();
    const double total_conc =
        std::accumulate(grid.cells().begin(), grid.cells().end(), 0.0);
    EXPECT_DOUBLE_EQ(total_conc, 5.0);  // unit cell volume
}

TEST(ConcentrationGrid, RejectsInvalidGeometry) {
    EXPECT_THROW(ConcentrationGrid({0, 0, 0}, {0.0, 1.0, 1.0}, {2, 2, 2}),
                 std::invalid_argument);
    EXPECT_THROW(ConcentrationGrid({0, 0, 0}, {1.0, 1.0, 1.0}, {0, 2, 2}),
                 std::invalid_argument);
}

TEST(EngineConcentration, TotalMassMatchesSourceMinusDecay) {
    auto meteo = std::make_shared<UniformMeteoField>(
        std::array<double, 3>{1.0, 0.0, 0.0},
        std::array<double, 3>{0.3, 0.3, 0.3}, 0.1);

    EngineConfig cfg;
    cfg.num_particles = 50000;
    cfg.dt = 0.1;
    cfg.source_pos = {5.0, 5.0, 5.0};
    cfg.source_mass = 10.0;
    cfg.decay_rate = 0.0;
    cfg.seed = 2024;

    LagrangianDispersionEngine engine(cfg, meteo);
    engine.advance(5);

    // Large grid that contains all particles after only 5 steps.
    const std::array<double, 3> cell{1.0, 1.0, 1.0};
    const std::array<std::size_t, 3> dims{200, 200, 200};
    const std::array<double, 3> origin{-50.0, -50.0, -50.0};
    const auto conc = engine.concentration(origin, cell, dims);

    const double cell_vol = cell[0] * cell[1] * cell[2];
    double total_mass = 0.0;
    for (double c : conc) {
        total_mass += c * cell_vol;
    }
    EXPECT_NEAR(total_mass, cfg.source_mass, 1e-6);
}
