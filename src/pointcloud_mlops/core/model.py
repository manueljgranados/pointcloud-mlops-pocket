from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def build_model(name: str, params: dict) -> object:
    if name == "random_forest":
        return RandomForestClassifier(**params)
    raise ValueError(f"Unknown model name: {name}")
