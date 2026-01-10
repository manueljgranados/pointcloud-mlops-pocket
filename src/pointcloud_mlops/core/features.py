from __future__ import annotations

import numpy as np


def _safe_norm(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return np.linalg.norm(x, axis=axis) + 1e-12


def extract_features(points: np.ndarray) -> np.ndarray:
    """
    points: (n,3)
    devuelve vector 1D con descriptores geométricos.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n,3)")
    n = points.shape[0]
    if n < 16:
        raise ValueError("need at least 16 points")

    # Centrar
    centroid = points.mean(axis=0, keepdims=True)
    centered = points - centroid

    # Distancias al centro
    d = _safe_norm(centered, axis=1)
    d_mean = d.mean()
    d_std = d.std()
    d_min = d.min()
    d_max = d.max()
    d_q25, d_q50, d_q75 = np.quantile(d, [0.25, 0.50, 0.75])

    # Bounding box
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = maxs - mins
    bbox_vol = float(spans[0] * spans[1] * spans[2])
    bbox_diag = float(np.linalg.norm(spans) + 1e-12)

    # Covarianza y PCA (autovalores)
    cov = np.cov(centered.T)
    evals = np.linalg.eigvalsh(cov)  # orden ascendente
    evals = np.maximum(evals, 1e-12)
    e1, e2, e3 = evals  # e3 mayor

    # Ratios PCA (invariantes a rotación)
    pca_sum = e1 + e2 + e3
    r1 = e1 / pca_sum
    r2 = e2 / pca_sum
    r3 = e3 / pca_sum
    # “Sphericity” simple: cercano a 1 si evals parecidos
    sphericity = float(e1 / e3)

    # Planaridad / linealidad (heurísticas)
    linearity = float((e3 - e2) / e3)
    planarity = float((e2 - e1) / e3)

    feats = np.array(
        [
            d_mean,
            d_std,
            d_min,
            d_max,
            d_q25,
            d_q50,
            d_q75,
            spans[0],
            spans[1],
            spans[2],
            bbox_vol,
            bbox_diag,
            e1,
            e2,
            e3,
            r1,
            r2,
            r3,
            sphericity,
            linearity,
            planarity,
        ],
        dtype=float,
    )
    return feats


def extract_features_batch(clouds: np.ndarray) -> np.ndarray:
    """
    clouds: (N, n_points, 3) -> (N, n_features)
    """
    if clouds.ndim != 3 or clouds.shape[2] != 3:
        raise ValueError("clouds must have shape (N, n_points, 3)")
    return np.vstack([extract_features(clouds[i]) for i in range(clouds.shape[0])])
