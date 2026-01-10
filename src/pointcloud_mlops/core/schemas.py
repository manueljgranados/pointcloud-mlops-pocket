from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Shape = Literal["sphere", "cube", "cylinder", "cone", "torus"]


class GenerateRequest(BaseModel):
    shape: Shape
    n_points: int = Field(default=512, ge=16, le=5000)
    noise_sigma: float = Field(default=0.02, ge=0.0, le=0.2)
    seed: int | None = None
    apply_random_rotation: bool = True


class GenerateResponse(BaseModel):
    shape: Shape
    n_points: int
    noise_sigma: float
    seed: int | None
    points: list[list[float]]  # [[x,y,z], ...]
    plotly_json: str
