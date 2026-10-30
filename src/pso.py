"""Particle swarm search over U-Net training hyperparameters."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


PARAMETER_NAMES = ("learning_rate", "dropout", "weight_decay")


@dataclass
class Particle:
    """A particle in log10(lr), dropout, log10(weight decay) space."""

    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray = field(init=False)
    best_fitness: float = field(default=math.inf)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        self.best_position = self.position.copy()


def bounds_from_config(config: dict) -> np.ndarray:
    """Convert configured physical bounds into the 3D search space."""
    pso = config["pso"]
    lr = pso["learning_rate_bounds"]
    dropout = pso["dropout_bounds"]
    decay = pso["weight_decay_bounds"]
    if not (0 < lr[0] < lr[1] and 0 <= dropout[0] < dropout[1] < 1
            and 0 < decay[0] < decay[1]):
        raise ValueError("Invalid PSO hyperparameter bounds")
    return np.array([
        [math.log10(lr[0]), math.log10(lr[1])],
        [dropout[0], dropout[1]],
        [math.log10(decay[0]), math.log10(decay[1])],
    ], dtype=float)


def decode_position(position: np.ndarray) -> dict[str, float]:
    """Decode particle coordinates to the model training parameters."""
    return {
        "learning_rate": float(10 ** position[0]),
        "dropout": float(position[1]),
        "weight_decay": float(10 ** position[2]),
    }


class ParticleSwarmOptimizer:
    """Minimize 1 - validation Dice with a fixed candidate evaluation budget."""

    def __init__(
        self,
        bounds: np.ndarray,
        particles: int = 3,
        evaluation_rounds: int = 2,
        inertia: float = 0.7,
        cognitive: float = 1.5,
        social: float = 1.5,
        seed: int = 42,
    ) -> None:
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.shape != (3, 2) or np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("PSO bounds must have shape (3, 2), with min < max")
        if particles < 1 or evaluation_rounds < 1:
            raise ValueError("particles and evaluation_rounds must be positive")
        self.particle_count = particles
        self.evaluation_rounds = evaluation_rounds
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.rng = np.random.default_rng(seed)
        positions = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], (particles, 3))
        self.particles = [Particle(position, np.zeros(3)) for position in positions]
        self.global_best_position: np.ndarray | None = None
        self.global_best_fitness = math.inf
        self.history: list[dict[str, float | int]] = []

    def observe(self, particle_index: int, validation_dice: float) -> float:
        """Update the personal and global best from a measured validation Dice."""
        if not math.isfinite(validation_dice) or not (0 <= validation_dice <= 1):
            raise ValueError("Candidate validation Dice must be finite and in [0, 1]")
        fitness = 1.0 - validation_dice
        particle = self.particles[particle_index]
        if fitness < particle.best_fitness:
            particle.best_fitness = fitness
            particle.best_position = particle.position.copy()
        if fitness < self.global_best_fitness:
            self.global_best_fitness = fitness
            self.global_best_position = particle.position.copy()
        return fitness

    def step(self) -> None:
        """Update all velocities and positions using independent random vectors."""
        if self.global_best_position is None:
            raise RuntimeError("Evaluate the initial swarm before updating positions")
        for particle in self.particles:
            r1, r2 = self.rng.random(3), self.rng.random(3)
            particle.velocity = (
                self.inertia * particle.velocity
                + self.cognitive * r1 * (particle.best_position - particle.position)
                + self.social * r2 * (self.global_best_position - particle.position)
            )
            proposed = particle.position + particle.velocity
            particle.position = np.clip(proposed, self.bounds[:, 0], self.bounds[:, 1])
            hit_boundary = proposed != particle.position
            particle.velocity[hit_boundary] = 0.0

    def optimize(self, objective: Callable[[dict[str, float], int], float]) -> dict:
        """Evaluate each particle once per round; objective returns validation Dice."""
        evaluation = 0
        started = time.perf_counter()
        for round_index in range(self.evaluation_rounds):
            if round_index:
                self.step()
            for particle_index, particle in enumerate(self.particles):
                parameters = decode_position(particle.position)
                candidate_started = time.perf_counter()
                validation_dice = float(objective(parameters, evaluation))
                candidate_seconds = time.perf_counter() - candidate_started
                fitness = self.observe(particle_index, validation_dice)
                self.history.append({
                    "round": round_index + 1,
                    "particle": particle_index + 1,
                    "evaluation": evaluation + 1,
                    **parameters,
                    "validation_dice": validation_dice,
                    "fitness": fitness,
                    "global_best_dice": 1.0 - self.global_best_fitness,
                    "candidate_seconds": candidate_seconds,
                })
                evaluation += 1
        assert self.global_best_position is not None
        return {
            "best_parameters": decode_position(self.global_best_position),
            "best_candidate_validation_dice": 1.0 - self.global_best_fitness,
            "evaluations": evaluation,
            "search_seconds": time.perf_counter() - started,
            "history": self.history,
        }


def write_pso_history(path: str | Path, history: list[dict]) -> None:
    """Save measured candidate outcomes as a CSV file."""
    if not history:
        raise ValueError("Cannot write empty PSO history")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
