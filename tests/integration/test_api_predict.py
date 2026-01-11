import importlib
import os

import numpy as np
from fastapi.testclient import TestClient

from pointcloud_mlops.adapters.io.artifact_store import save_model_artifacts
from pointcloud_mlops.adapters.io.dataset_loader import make_synthetic_dataset, split_indices
from pointcloud_mlops.core.features import extract_features_batch
from pointcloud_mlops.core.model import build_model
from pointcloud_mlops.core.pointcloud_generator import GenerateParams


def _train_tiny_model(artifacts_dir: str):
    classes = ["sphere", "cube", "cylinder", "cone", "torus"]
    gen_base = GenerateParams(shape="sphere", n_points=128, noise_sigma=0.02, seed=None)
    clouds, y, fp = make_synthetic_dataset(
        classes=classes, n_samples_per_class=10, gen_base=gen_base, seed=123
    )
    tr, va, te = split_indices(clouds.shape[0], train=0.7, val=0.15, test=0.15, seed=123)
    X = extract_features_batch(clouds)
    model = build_model("random_forest", {"n_estimators": 50, "random_state": 123, "n_jobs": -1})
    model.fit(X[tr], y[tr])

    meta = {
        "classes": classes,
        "dataset_fingerprint": fp,
        "metrics": {},
        "model_version": "SET_BY_ARTIFACT_STORE",
    }
    saved = save_model_artifacts(artifacts_dir=artifacts_dir, model=model, metadata=meta)
    return saved


def test_predict_works(tmp_path):
    _train_tiny_model(str(tmp_path))

    os.environ["MODEL_DIR"] = str(tmp_path / "models" / "current")

    import pointcloud_mlops.main_api as main_api

    importlib.reload(main_api)

    # Generar puntos simples para pedir predicción (no importa la clase exacta aquí)
    pts = np.random.normal(size=(256, 3)).tolist()

    with TestClient(main_api.app) as client:
        r = client.post("/predict", json={"points": pts})
    assert r.status_code == 200
    data = r.json()
    assert "predicted_class" in data
    assert "probabilities" in data
    assert "plotly_json" in data
