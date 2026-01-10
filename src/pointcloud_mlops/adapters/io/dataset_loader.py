from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

import numpy as np

from pointcloud_mlops.core.pointcloud_generator import GenerateParams, generate_pointcloud


def _fingerprint(payload: dict[str, Any]) -> str:
    b = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:16]


def make_synthetic_dataset(
    *,
    classes: list[str],
    n_samples_per_class: int,
    gen_base: GenerateParams,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Devuelve:
      - clouds: (N, n_points, 3)
      - y: (N,)
      - dataset_fingerprint: str
    """
    rng = np.random.default_rng(seed)
    total = len(classes) * n_samples_per_class
    clouds = np.empty((total, gen_base.n_points, 3), dtype=float)
    y = np.empty((total,), dtype=int)

    idx = 0
    for label, shape in enumerate(classes):
        for _ in range(n_samples_per_class):
            # Semilla por muestra para reproducibilidad
            sample_seed = int(rng.integers(0, 2**31 - 1))
            params = GenerateParams(**{**asdict(gen_base), "shape": shape, "seed": sample_seed})
            clouds[idx] = generate_pointcloud(params)
            y[idx] = label
            idx += 1

    fp_payload = {
        "classes": classes,
        "n_samples_per_class": n_samples_per_class,
        "gen_base": asdict(gen_base),
        "seed": seed,
        "total_samples": int(total),
    }
    return clouds, y, _fingerprint(fp_payload)


def split_indices(
    n: int, *, train: float, val: float, test: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(train + val + test, 1.0):
        raise ValueError("train+val+test must sum to 1.0")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(round(n * train))
    n_val = int(round(n * val))
    # resto a test
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx
