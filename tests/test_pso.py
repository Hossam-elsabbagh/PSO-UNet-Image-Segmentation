"""Deterministic unit checks for the PSO search mechanics."""

import numpy as np
import pytest

from src.pso import ParticleSwarmOptimizer, bounds_from_config, decode_position


CONFIG = {"pso": {
    "learning_rate_bounds": [0.0001, 0.003],
    "dropout_bounds": [0.0, 0.3],
    "weight_decay_bounds": [0.000001, 0.001],
}}


def test_initialization_and_decode_are_bounded_and_deterministic():
    bounds = bounds_from_config(CONFIG)
    left = ParticleSwarmOptimizer(bounds, particles=3, seed=42)
    right = ParticleSwarmOptimizer(bounds, particles=3, seed=42)
    for a, b in zip(left.particles, right.particles):
        np.testing.assert_array_equal(a.position, b.position)
        assert np.all(a.position >= bounds[:, 0])
        assert np.all(a.position <= bounds[:, 1])
        decoded = decode_position(a.position)
        assert 0.0001 <= decoded["learning_rate"] <= 0.003
        assert 0 <= decoded["dropout"] <= 0.3
        assert 0.000001 <= decoded["weight_decay"] <= 0.001


def test_personal_and_global_best_only_update_for_improvement():
    optimizer = ParticleSwarmOptimizer(bounds_from_config(CONFIG), particles=2)
    initial = optimizer.particles[0].position.copy()
    optimizer.observe(0, 0.4)
    optimizer.particles[0].position += 0.01
    optimizer.observe(0, 0.2)
    np.testing.assert_array_equal(optimizer.particles[0].best_position, initial)
    optimizer.observe(1, 0.6)
    assert optimizer.global_best_fitness == pytest.approx(0.4)
    np.testing.assert_array_equal(optimizer.global_best_position, optimizer.particles[1].position)


def test_velocity_step_enforces_position_bounds():
    optimizer = ParticleSwarmOptimizer(bounds_from_config(CONFIG), particles=2)
    optimizer.observe(0, 0.5)
    optimizer.observe(1, 0.6)
    optimizer.particles[0].velocity[:] = 100
    optimizer.step()
    for particle in optimizer.particles:
        assert np.all(particle.position >= optimizer.bounds[:, 0])
        assert np.all(particle.position <= optimizer.bounds[:, 1])


def test_fixed_seed_reproduces_full_search():
    def search():
        optimizer = ParticleSwarmOptimizer(bounds_from_config(CONFIG), particles=3, evaluation_rounds=2, seed=42)
        return optimizer.optimize(lambda params, _: 0.5 + params["dropout"] / 2)

    first, second = search(), search()
    assert first["evaluations"] == 6
    assert first["best_parameters"] == second["best_parameters"]
    assert [row["validation_dice"] for row in first["history"]] == [
        row["validation_dice"] for row in second["history"]
    ]


def test_invalid_dice_rejected():
    optimizer = ParticleSwarmOptimizer(bounds_from_config(CONFIG))
    with pytest.raises(ValueError):
        optimizer.observe(0, float("nan"))
