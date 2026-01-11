from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from pointcloud_mlops.adapters.io.artifact_store import save_model_artifacts
from pointcloud_mlops.adapters.io.dataset_loader import make_synthetic_dataset, split_indices
from pointcloud_mlops.core.features import extract_features_batch
from pointcloud_mlops.core.metrics import compute_metrics
from pointcloud_mlops.core.model import build_model
from pointcloud_mlops.core.pointcloud_generator import GenerateParams


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    seed = int(cfg["seed"])
    classes = list(cfg["dataset"]["classes"])

    gen_base = GenerateParams(
        shape="sphere",  # se sobreescribe por clase
        n_points=int(cfg["dataset"]["n_points"]),
        noise_sigma=float(cfg["dataset"]["noise_sigma"]),
        seed=None,
        apply_random_rotation=bool(cfg["dataset"].get("apply_random_rotation", True)),
        scale_range=tuple(cfg["dataset"].get("scale_range", [0.85, 1.15])),
        translation_sigma=float(cfg["dataset"].get("translation_sigma", 0.02)),
    )

    clouds, y, dataset_fp = make_synthetic_dataset(
        classes=classes,
        n_samples_per_class=int(cfg["dataset"]["n_samples_per_class"]),
        gen_base=gen_base,
        seed=seed,
    )

    tr_idx, va_idx, te_idx = split_indices(
        clouds.shape[0],
        train=float(cfg["split"]["train"]),
        val=float(cfg["split"]["val"]),
        test=float(cfg["split"]["test"]),
        seed=seed,
    )

    X = extract_features_batch(clouds)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    model = build_model(cfg["model"]["name"], dict(cfg["model"]["params"]))
    model.fit(X_tr, y_tr)

    pred_va = model.predict(X_va)
    pred_te = model.predict(X_te)

    m_va = compute_metrics(y_va, pred_va)
    m_te = compute_metrics(y_te, pred_te)

    metadata = {
        "project": "pointcloud-mlops-pocket",
        "classes": classes,
        "dataset_fingerprint": dataset_fp,
        "config_used": cfg,
        "metrics": {"val": m_va, "test": m_te},
    }

    saved = save_model_artifacts(
        artifacts_dir=str(cfg["artifacts_dir"]),
        model=model,
        metadata={**metadata, "model_version": "SET_BY_ARTIFACT_STORE"},
    )

    # Actualizar metadata con versión real
    meta_path = saved.model_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["model_version"] = saved.version
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # También actualizar current/metadata.json
    current_meta = Path(cfg["artifacts_dir"]) / "models" / "current" / "metadata.json"
    current_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Modelo guardado. version={saved.version}")
    print("Métricas val:", m_va)
    print("Métricas test:", m_te)

    report = {
        "model_version": saved.version,
        "dataset_fingerprint": dataset_fp,
        "metrics": {"val": m_va, "test": m_te},
        "classes": classes,
    }
    reports_dir = Path(cfg["artifacts_dir"]) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
