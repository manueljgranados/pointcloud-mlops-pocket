import os

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pointcloud_mlops.adapters.io.artifact_store import load_model
from pointcloud_mlops.core.features import extract_features
from pointcloud_mlops.core.plotly_viz import figure_to_json, pointcloud_figure
from pointcloud_mlops.core.pointcloud_generator import GenerateParams, generate_pointcloud
from pointcloud_mlops.core.schemas import GenerateRequest, GenerateResponse

app = FastAPI(title="PointCloud 3D - Mini MLOps")

MODEL_DIR = os.getenv("MODEL_DIR", "artifacts/models/current")


class PredictRequest(BaseModel):
    points: list[list[float]] = Field(..., min_length=16, max_length=5000)


class PredictResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
    model_version: str | None
    plotly_json: str


@app.get("/health")
def health():
    # Si no hay modelo aún, model_version = None
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    model_version = None
    if os.path.exists(meta_path):
        import json

        model_version = json.loads(open(meta_path, encoding="utf-8").read()).get("model_version")
    return {"status": "ok", "model_version": model_version}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        pts = generate_pointcloud(
            GenerateParams(
                shape=req.shape,
                n_points=req.n_points,
                noise_sigma=req.noise_sigma,
                seed=req.seed,
                apply_random_rotation=req.apply_random_rotation,
            )
        )
        fig = pointcloud_figure(pts, title=f"Generated: {req.shape}")
        return GenerateResponse(
            shape=req.shape,
            n_points=req.n_points,
            noise_sigma=req.noise_sigma,
            seed=req.seed,
            points=pts.tolist(),
            plotly_json=figure_to_json(fig),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        model, meta = load_model(MODEL_DIR)
    except Exception as err:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not available. Train first and ensure MODEL_DIR points to "
                "artifacts/models/current."
            ),
        ) from err

    pts = np.asarray(req.points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise HTTPException(status_code=400, detail="points must be an array of shape (n,3)")

    x = extract_features(pts).reshape(1, -1)
    proba = model.predict_proba(x)[0]
    pred_idx = int(np.argmax(proba))
    classes = meta.get("classes", [])
    if not classes:
        raise HTTPException(status_code=500, detail="Model metadata missing classes")

    pred_class = classes[pred_idx]
    probabilities = {classes[i]: float(proba[i]) for i in range(len(classes))}

    fig = pointcloud_figure(pts, title=f"Predicted: {pred_class}")
    return PredictResponse(
        predicted_class=pred_class,
        probabilities=probabilities,
        model_version=meta.get("model_version"),
        plotly_json=figure_to_json(fig),
    )
