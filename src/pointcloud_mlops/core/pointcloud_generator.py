from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Shape = Literal["sphere", "cube", "cylinder", "cone", "torus"]


@dataclass(frozen=True)
class GenerateParams:
    shape: Shape
    n_points: int = 512
    noise_sigma: float = 0.02
    seed: int | None = None
    apply_random_rotation: bool = True
    scale_range: tuple[float, float] = (0.85, 1.15)
    translation_sigma: float = 0.02


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    # Rotación uniforme en SO(3) via cuaterniones
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q0 = np.sqrt(u1) * np.cos(2 * np.pi * u3)

    # Matriz de rotación desde cuaternión (q0 escalar)
    r00 = 1 - 2 * (q2**2 + q3**2)
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 1 - 2 * (q1**2 + q3**2)
    r12 = 2 * (q2 * q3 - q0 * q1)

    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 1 - 2 * (q1**2 + q2**2)

    return np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], dtype=float)


def _sample_sphere_surface(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v  # radio 1


def _sample_cube_surface(rng: np.random.Generator, n: int) -> np.ndarray:
    # Cubo en [-1,1]^3, muestreando caras
    pts = np.empty((n, 3), dtype=float)
    face_axis = rng.integers(0, 3, size=n)
    face_sign = rng.choice([-1.0, 1.0], size=n)

    pts[:, 0] = rng.uniform(-1, 1, size=n)
    pts[:, 1] = rng.uniform(-1, 1, size=n)
    pts[:, 2] = rng.uniform(-1, 1, size=n)

    for ax in (0, 1, 2):
        mask = face_axis == ax
        pts[mask, ax] = face_sign[mask]
    return pts


def _sample_cylinder_surface(rng: np.random.Generator, n: int) -> np.ndarray:
    # Radio 1, altura 2 (z en [-1,1]); mezcla lateral + tapas
    pts = np.empty((n, 3), dtype=float)
    is_lateral = rng.random(n) < 0.7

    # Lateral
    nl = int(is_lateral.sum())
    theta = rng.uniform(0, 2 * np.pi, size=nl)
    z = rng.uniform(-1, 1, size=nl)
    pts[is_lateral, 0] = np.cos(theta)
    pts[is_lateral, 1] = np.sin(theta)
    pts[is_lateral, 2] = z

    # Tapas
    nc = n - nl
    theta2 = rng.uniform(0, 2 * np.pi, size=nc)
    r = np.sqrt(rng.random(nc))
    x = r * np.cos(theta2)
    y = r * np.sin(theta2)
    zcap = rng.choice([-1.0, 1.0], size=nc)
    pts[~is_lateral, 0] = x
    pts[~is_lateral, 1] = y
    pts[~is_lateral, 2] = zcap
    return pts


def _sample_cone_surface(rng: np.random.Generator, n: int) -> np.ndarray:
    # Cono con ápice en z=1 y base en z=-1 (radio 1)
    pts = np.empty((n, 3), dtype=float)
    is_lateral = rng.random(n) < 0.7

    nl = int(is_lateral.sum())
    t = rng.random(nl)  # 0..1 desde ápice a base
    theta = rng.uniform(0, 2 * np.pi, size=nl)
    radius = t * 1.0
    z = 1.0 - 2.0 * t
    pts[is_lateral, 0] = radius * np.cos(theta)
    pts[is_lateral, 1] = radius * np.sin(theta)
    pts[is_lateral, 2] = z

    nc = n - nl
    theta2 = rng.uniform(0, 2 * np.pi, size=nc)
    r = np.sqrt(rng.random(nc))
    pts[~is_lateral, 0] = r * np.cos(theta2)
    pts[~is_lateral, 1] = r * np.sin(theta2)
    pts[~is_lateral, 2] = -1.0
    return pts


def _sample_torus_surface(rng: np.random.Generator, n: int) -> np.ndarray:
    # Toro en torno a eje z: (R + r cos v)cos u, ...; normalizado para encajar aprox. en [-1,1]
    R, r = 1.2, 0.4
    u = rng.uniform(0, 2 * np.pi, size=n)
    v = rng.uniform(0, 2 * np.pi, size=n)

    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    pts = np.stack([x, y, z], axis=1)
    pts = pts / (R + r)  # escala para que x,y queden ~[-1,1]
    return pts


def generate_pointcloud(params: GenerateParams) -> np.ndarray:
    if params.n_points <= 0:
        raise ValueError("n_points must be > 0")

    rng = np.random.default_rng(params.seed)

    if params.shape == "sphere":
        pts = _sample_sphere_surface(rng, params.n_points)
    elif params.shape == "cube":
        pts = _sample_cube_surface(rng, params.n_points)
    elif params.shape == "cylinder":
        pts = _sample_cylinder_surface(rng, params.n_points)
    elif params.shape == "cone":
        pts = _sample_cone_surface(rng, params.n_points)
    elif params.shape == "torus":
        pts = _sample_torus_surface(rng, params.n_points)
    else:
        raise ValueError(f"Unknown shape: {params.shape}")

    # Transformaciones para que el modelo no memorice una plantilla fija
    if params.apply_random_rotation:
        rot = _random_rotation_matrix(rng)
        pts = pts @ rot.T

    smin, smax = params.scale_range
    if smin > 0 and smax >= smin:
        scale = rng.uniform(smin, smax)
        pts = pts * scale

    if params.translation_sigma > 0:
        t = rng.normal(scale=params.translation_sigma, size=(1, 3))
        pts = pts + t

    if params.noise_sigma > 0:
        pts = pts + rng.normal(scale=params.noise_sigma, size=pts.shape)

    return pts.astype(float)
