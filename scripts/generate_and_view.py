from __future__ import annotations

import argparse
from pathlib import Path

from pointcloud_mlops.core.plotly_viz import figure_to_html, pointcloud_figure
from pointcloud_mlops.core.pointcloud_generator import GenerateParams, generate_pointcloud


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shape", required=True, choices=["sphere", "cube", "cylinder", "cone", "torus"]
    )
    p.add_argument("--n_points", type=int, default=512)
    p.add_argument("--noise_sigma", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="artifacts/reports/preview.html")
    args = p.parse_args()

    pts = generate_pointcloud(
        GenerateParams(
            shape=args.shape,
            n_points=args.n_points,
            noise_sigma=args.noise_sigma,
            seed=args.seed,
        )
    )
    fig = pointcloud_figure(pts, title=f"Preview: {args.shape}")
    html = figure_to_html(fig)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    print(f"HTML generado: {out_path.resolve()}")
    print("Ábralo en el navegador.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
