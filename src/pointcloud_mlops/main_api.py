from __future__ import annotations

import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from pointcloud_mlops.core.plotly_viz import figure_to_json, pointcloud_figure
from pointcloud_mlops.core.pointcloud_generator import GenerateParams, generate_pointcloud
from pointcloud_mlops.core.schemas import GenerateRequest, GenerateResponse
from pointcloud_mlops.services.model_runtime import ModelRuntime

app = FastAPI(title="PointCloud 3D - Mini MLOps")


def get_model_dir() -> str:
    return os.getenv("MODEL_DIR", "artifacts/models/current")


@app.on_event("startup")
def _startup_load_model() -> None:
    app.state.runtime = ModelRuntime()
    model_dir = get_model_dir()
    try:
        app.state.runtime.load(model_dir)
    except Exception:
        # Es válido arrancar sin modelo; /predict devolverá 503.
        pass


class PredictRequest(BaseModel):
    points: list[list[float]] = Field(..., min_length=16, max_length=5000)


class PredictResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
    model_version: str | None
    plotly_json: str


@app.get("/health")
def health():
    rt: ModelRuntime = app.state.runtime
    return {"status": "ok", "model_loaded": rt.is_loaded, "model_version": rt.version}


@app.get("/demo", response_class=HTMLResponse)
def demo():
    # Página mínima: generar -> ver -> predecir
    return HTMLResponse(
        """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <title>PointCloud Demo</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
    .row{ display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
    #plot{ width: 100%; height: 70vh; border: 1px solid #ddd; border-radius: 10px; }
    code{ background:#f5f5f5; padding:2px 6px; border-radius:6px; }
  </style>
</head>
<body>
  <h2>PointCloud 3D — Demo</h2>

  <div class="row">
    <label>Forma:
      <select id="shape">
        <option value="sphere">sphere</option>
        <option value="cube">cube</option>
        <option value="cylinder">cylinder</option>
        <option value="cone">cone</option>
        <option value="torus">torus</option>
      </select>
    </label>

    <label>Ruido (sigma):
      <input id="sigma" type="number" step="0.01" value="0.02" min="0" max="0.2"/>
    </label>

    <label>Seed:
      <input id="seed" type="number" step="1" value="123"/>
    </label>

    <button id="btnGen">Generar</button>
    <button id="btnPred">Predecir</button>

    <span id="status"></span>
  </div>

  <div id="plot"></div>

  <p>
    Endpoints: <code>POST /generate</code> y <code>POST /predict</code>.
  </p>

<script>
let lastPoints = null;

async function renderPlotly(figJson){
  try{
    const fig = JSON.parse(figJson);
    await Plotly.newPlot("plot", fig.data, fig.layout, {responsive:true});
  }catch(e){
    console.error("Plotly render error:", e);
    document.getElementById("status").textContent = "Error renderizando Plotly. Revise consola.";
  }
}

function renderFromPoints(points, title){
  const x = points.map(p => p[0]);
  const y = points.map(p => p[1]);
  const z = points.map(p => p[2]);

  return Plotly.newPlot("plot", [{
    type: "scatter3d",
    mode: "markers",
    x, y, z,
    marker: { size: 2, opacity: 0.8 }
  }], {
    title,
    scene: { aspectmode: "data" },
    margin: { l:0, r:0, t:40, b:0 },
    showlegend: false
  }, { responsive: true });
}


async function postJson(url, payload){
  const res = await fetch(url, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload)
  });
  if(!res.ok){
    const txt = await res.text();
    throw new Error(txt);
  }
  return await res.json();
}

document.getElementById("btnGen").onclick = async () => {
  try{
    document.getElementById("status").textContent = "Generando...";
    const shape = document.getElementById("shape").value;
    const noise_sigma = parseFloat(document.getElementById("sigma").value);
    const seed = parseInt(document.getElementById("seed").value);

    const out = await postJson("/generate", {
      shape, n_points: 512, noise_sigma, seed, apply_random_rotation: true
    });

    lastPoints = out.points;
    await renderFromPoints(out.points, `Generated: ${shape}`);
    document.getElementById("status").textContent = "Nube generada.";
  }catch(e){
    document.getElementById("status").textContent = "Error: " + e.message;
  }
};

document.getElementById("btnPred").onclick = async () => {
  try{
    if(!lastPoints){
      alert("Primero genere una nube.");
      return;
    }
    document.getElementById("status").textContent = "Prediciendo...";
    const out = await postJson("/predict", {points: lastPoints});
    await renderFromPoints(lastPoints, `Predicted: ${out.predicted_class}`);
    document.getElementById("status").textContent =
      `Predicción: ${out.predicted_class} (modelo: ${out.model_version ?? "N/A"})`;
  }catch(e){
    document.getElementById("status").textContent = "Error: " + e.message;
  }
};
</script>
</body>
</html>
"""
    )


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
    rt: ModelRuntime = app.state.runtime
    if not rt.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train first and ensure MODEL_DIR points to "
            "artifacts/models/current.",
        )

    pts = np.asarray(req.points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise HTTPException(status_code=400, detail="points must have shape (n,3)")

    pred_class, probabilities = rt.predict(pts)
    fig = pointcloud_figure(pts, title=f"Predicted: {pred_class}")

    return PredictResponse(
        predicted_class=pred_class,
        probabilities=probabilities,
        model_version=rt.version,
        plotly_json=figure_to_json(fig),
    )
