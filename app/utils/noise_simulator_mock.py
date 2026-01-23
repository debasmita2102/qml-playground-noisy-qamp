"""
Utilities for generating synthetic quantum state trajectories with ideal and noisy
Bloch vector evolutions. This module provides deterministic mock data that mimics
how single-qubit states would evolve under different noise channels. The data can
be used to drive visualizations without invoking real quantum simulations.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Literal, Tuple

import numpy as np


# Reasonable defaults for visualization granularity
DEFAULT_NUM_STEPS = 60

NoiseType = Literal["Depolarizing", "Amplitude", "Phase"]

logger = logging.getLogger(" [MOCK NOISE]")

_NOISE_DESCRIPTIONS = {
    "depolarizing": (
        "Uniformly shrinks every Bloch vector towards the origin and sprinkles a small "
        "random jitter so the path does not collapse onto a straight line."
    ),
    "amplitude": (
        "Damps excitations so the trajectory migrates towards |0⟩. The transverse "
        "components (x,y) shrink with √(1-γ) while the z-axis is biased upwards."
    ),
    "phase": (
        "Suppresses phase coherence: x and y contract by (1-γ) and a bit of noise is "
        "added, while the population along z is kept largely intact."
    ),
}


def generate_mock_trajectories(
    noise_type: NoiseType = "Depolarizing",
    depolarizing_probability: float = 0.1,
    damping_rate: float = 0.2,
    num_steps: int = DEFAULT_NUM_STEPS,
) -> Dict[str, List[List[float]]]:
    """
    Create an ideal and noisy Bloch vector trajectory for a single qubit.

    Args:
        noise_type: Noise channel to mimic. Supported values: "Depolarizing",
            "Amplitude", and "Phase".
        depolarizing_probability: Probability parameter for depolarizing noise.
            Only used when ``noise_type`` is ``"Depolarizing"``.
        damping_rate: Damping rate ``gamma`` for amplitude/phase damping channels.
        num_steps: Number of time steps in the trajectory.

    Returns:
        Dictionary containing ideal and noisy Bloch vectors as well as metadata
        that describes the generated mock trajectory.
    """
    logger.info("=== Mock noise trajectory request ===")
    logger.info(
        "Parameters -> noise=%s | dep=%.3f | gamma=%.3f | steps=%d",
        noise_type,
        depolarizing_probability,
        damping_rate,
        num_steps,
    )

    trajectory, time_points = _generate_ideal_trajectory(num_steps)
    _log_vector_snapshot("Step 1 - Ideal path", trajectory)

    rng = _parameter_seeded_rng(noise_type, depolarizing_probability, damping_rate)

    noise_type_lower = noise_type.lower()
    if noise_type_lower == "depolarizing":
        noisy_traj, detail = _apply_depolarizing_noise(
            trajectory, depolarizing_probability, rng
        )
    elif noise_type_lower == "amplitude":
        noisy_traj, detail = _apply_amplitude_damping(trajectory, damping_rate)
    elif noise_type_lower == "phase":
        noisy_traj, detail = _apply_phase_damping(trajectory, damping_rate, rng)
    else:
        raise ValueError(
            f"Unsupported noise type '{noise_type}'. "
            "Choose from: Depolarizing, Amplitude, Phase."
        )

    logger.info(
        "Step 2 - %s noise applied (%s)",
        noise_type.capitalize(),
        _format_detail(detail),
    )
    logger.info("         %s", _NOISE_DESCRIPTIONS.get(noise_type_lower, ""))

    _log_vector_snapshot("Step 3 - Noisy path", noisy_traj)
    _log_alignment_metrics(trajectory, noisy_traj)

    meta = {
        "noise_type": noise_type,
        "detail": detail,
        "description": _NOISE_DESCRIPTIONS.get(noise_type_lower, ""),
    }

    return {
        "time": time_points.tolist(),
        "ideal": trajectory.tolist(),
        "noisy": noisy_traj.tolist(),
        "meta": meta,
    }


def _generate_ideal_trajectory(num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a smoothly varying Bloch vector trajectory."""
    t_values = np.linspace(0.0, 1.0, num_steps)
    traj = np.zeros((num_steps, 3), dtype=float)

    for idx, t in enumerate(t_values):
        # Use a gently precessing path around the Bloch sphere
        theta = math.pi * (0.25 + 0.5 * math.sin(2 * math.pi * t))
        phi = 2 * math.pi * t
        traj[idx] = [
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ]

    return traj, t_values


def _apply_depolarizing_noise(
    trajectory: np.ndarray,
    probability: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Depolarizing noise shrinks the Bloch vector towards the origin.

    The Bloch vector is contracted by a factor (1 - 4p/3). A small amount of
    parameter-dependent jitter is added to make the motion visually distinct.
    """
    probability = float(np.clip(probability, 0.0, 0.5))
    shrink = max(0.0, 1.0 - (4.0 * probability / 3.0))
    jitter_scale = 0.04 + 0.08 * probability

    noisy = shrink * trajectory
    noisy += rng.normal(loc=0.0, scale=jitter_scale, size=trajectory.shape)
    noisy = _normalize_vectors(noisy)

    detail = {
        "probability": probability,
        "shrink_factor": shrink,
        "jitter_sigma": jitter_scale,
    }
    return noisy, detail


def _apply_amplitude_damping(
    trajectory: np.ndarray,
    gamma: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Amplitude damping drives the state towards |0⟩ (north pole).

    Mapping for Bloch components:
        x -> sqrt(1 - gamma) * x
        y -> sqrt(1 - gamma) * y
        z -> (1 - gamma) * z + gamma
    """
    gamma = float(np.clip(gamma, 0.0, 1.0))
    decay = math.sqrt(max(0.0, 1.0 - gamma))

    noisy = np.empty_like(trajectory)
    noisy[:, 0] = decay * trajectory[:, 0]
    noisy[:, 1] = decay * trajectory[:, 1]
    noisy[:, 2] = (1.0 - gamma) * trajectory[:, 2] + gamma

    noisy = _normalize_vectors(noisy)
    detail = {
        "gamma": gamma,
        "transverse_decay": decay,
        "z_bias": gamma,
    }
    return noisy, detail


def _apply_phase_damping(
    trajectory: np.ndarray,
    gamma: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Phase damping reduces coherences (x/y components) while keeping z intact.

    Mapping:
        x -> (1 - gamma) * x + jitter
        y -> (1 - gamma) * y + jitter
        z -> z
    """
    gamma = float(np.clip(gamma, 0.0, 1.0))
    coherence_scale = max(0.0, 1.0 - gamma)
    jitter_scale = 0.02 + 0.05 * gamma

    noisy = trajectory.copy()
    noisy[:, 0] = coherence_scale * noisy[:, 0]
    noisy[:, 1] = coherence_scale * noisy[:, 1]
    noisy += rng.normal(loc=0.0, scale=jitter_scale, size=trajectory.shape)

    # Preserve z-component dominance
    noisy[:, 2] = trajectory[:, 2] * (0.9 + 0.1 * coherence_scale)

    noisy = _normalize_vectors(noisy)
    detail = {
        "gamma": gamma,
        "coherence_scale": coherence_scale,
        "jitter_sigma": jitter_scale,
    }
    return noisy, detail


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize Bloch vectors to lie within the unit ball."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1.0)
    return vectors / norms


def _parameter_seeded_rng(
    noise_type: str, depolarizing_probability: float, damping_rate: float
) -> np.random.Generator:
    """Create a deterministic RNG based on the noise parameters."""
    scaled_depol = int(round(depolarizing_probability * 1000))
    scaled_gamma = int(round(damping_rate * 1000))
    type_sum = sum(ord(char) for char in noise_type.lower())
    seed = (scaled_depol * 1315423911 + scaled_gamma * 2654435761 + type_sum) & 0xFFFFFFFF
    return np.random.default_rng(seed or 1)


def _log_vector_snapshot(label: str, vectors: np.ndarray) -> None:
    """Log start, midpoint, and end Bloch vectors for a trajectory."""
    if vectors.size == 0:
        logger.info("%s -> <empty>", label)
        return
    mid_index = len(vectors) // 2
    logger.info(
        "%s | start=%s | mid=%s | end=%s",
        label,
        _format_vector(vectors[0]),
        _format_vector(vectors[mid_index]),
        _format_vector(vectors[-1]),
    )


def _log_alignment_metrics(ideal: np.ndarray, noisy: np.ndarray) -> None:
    """Log simple deviation metrics between ideal and noisy paths."""
    delta = np.linalg.norm(ideal - noisy, axis=1)
    logger.info(
        "Deviation statistics | min=%.4f | mean=%.4f | max=%.4f",
        float(delta.min()),
        float(delta.mean()),
        float(delta.max()),
    )
    final_alignment = float(np.dot(ideal[-1], noisy[-1]))
    final_alignment = np.clip(final_alignment, -1.0, 1.0)
    logger.info(
        "Final-step alignment (ideal·noisy) = %.4f  =>  cosine angle ≈ %.2f°",
        final_alignment,
        math.degrees(math.acos(final_alignment)),
    )


def _format_vector(vector: np.ndarray) -> str:
    """Nicely format a 3D Bloch vector."""
    return "[" + ", ".join(f"{float(v):+.3f}" for v in vector) + "]"


def _format_detail(detail: Dict[str, float]) -> str:
    """Format a dictionary of detail values for logging."""
    parts = []
    for key, value in detail.items():
        if isinstance(value, (float, np.floating)):
            parts.append(f"{key}={float(value):.4f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


__all__ = ["generate_mock_trajectories", "DEFAULT_NUM_STEPS", "NoiseType"]

