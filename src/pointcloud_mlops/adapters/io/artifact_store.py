from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib


@dataclass(frozen=True)
class SavedModel:
    version: str
    model_dir: Path


def _now_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def save_model_artifacts(
    *,
    artifacts_dir: str,
    model: object,
    metadata: dict,
) -> SavedModel:
    base = Path(artifacts_dir)
    models_dir = base / "models"
    version = _now_version()
    out_dir = models_dir / version
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Actualizar "current" como copia (portable; evita symlinks problemáticos en Windows)
    current_dir = models_dir / "current"
    if current_dir.exists():
        shutil.rmtree(current_dir)
    shutil.copytree(out_dir, current_dir)

    return SavedModel(version=version, model_dir=out_dir)


def load_model(model_dir: str) -> tuple[object, dict]:
    p = Path(model_dir)
    model = joblib.load(p / "model.joblib")
    meta = json.loads((p / "metadata.json").read_text(encoding="utf-8"))
    return model, meta
