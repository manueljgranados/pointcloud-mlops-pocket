from fastapi import FastAPI, HTTPException

from pointcloud_mlops.core.plotly_viz import figure_to_json, pointcloud_figure
from pointcloud_mlops.core.pointcloud_generator import GenerateParams, generate_pointcloud
from pointcloud_mlops.core.schemas import GenerateRequest, GenerateResponse

app = FastAPI(title="PointCloud 3D - Mini MLOps")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": None}


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


@app.post("/predict")
def predict():
    return {"detail": "Not implemented yet. Train a model first."}
