from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pointcloud_mlops.adapters.io.artifact_store import load_model
from pointcloud_mlops.core.features import extract_features


@dataclass
class ModelRuntime:
    model: object | None = None
    meta: dict | None = None
    model_dir: str | None = None

    def load(self, model_dir: str) -> None:
        self.model, self.meta = load_model(model_dir)
        self.model_dir = model_dir

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.meta is not None

    @property
    def classes(self) -> list[str]:
        if not self.meta:
            return []
        return list(self.meta.get("classes", []))

    @property
    def version(self) -> str | None:
        if not self.meta:
            return None
        return self.meta.get("model_version")

    def predict(self, points: np.ndarray) -> tuple[str, dict[str, float]]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        x = extract_features(points).reshape(1, -1)
        proba = self.model.predict_proba(x)[0]  # type: ignore[attr-defined]

        classes = self.classes
        if not classes:
            raise RuntimeError("Missing classes in metadata")

        pred_idx = int(np.argmax(proba))
        pred_class = classes[pred_idx]
        probabilities = {classes[i]: float(proba[i]) for i in range(len(classes))}
        return pred_class, probabilities
