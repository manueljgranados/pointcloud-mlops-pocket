from fastapi import FastAPI

app = FastAPI(title="PointCloud 3D - Mini MLOps")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": None}


@app.post("/predict")
def predict():
    return {"detail": "Not implemented yet. Train a model first."}
