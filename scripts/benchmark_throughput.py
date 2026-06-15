#!/usr/bin/env python3
"""Benchmark Lagrangian engine throughput (particles x steps per second)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))

from nephele.reference import ReferenceDispersionEngine, UniformMeteo  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NEPHELE throughput benchmark")
    parser.add_argument("--particles", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args(argv)

    meteo = UniformMeteo(mean_wind=(3.0, 0.0, 0.0), sigma2=(0.4, 0.4, 0.4),
                         epsilon=0.1)
    engine = ReferenceDispersionEngine(
        meteo=meteo, num_particles=args.particles, dt=0.1,
        source_pos=(0.0, 0.0, 10.0), seed=1,
    )

    start = time.perf_counter()
    engine.advance(args.steps)
    elapsed = time.perf_counter() - start

    updates = args.particles * args.steps
    print(f"Particles        : {args.particles:,}")
    print(f"Steps            : {args.steps:,}")
    print(f"Elapsed          : {elapsed:.3f} s")
    print(f"Particle-updates : {updates / elapsed:,.0f} /s")
    print(f"Per-step latency : {1e3 * elapsed / args.steps:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
