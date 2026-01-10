import numpy as np

from pointcloud_mlops.core.features import extract_features
from pointcloud_mlops.core.pointcloud_generator import GenerateParams, generate_pointcloud


def test_extract_features_shape_and_finite():
    pts = generate_pointcloud(
        GenerateParams(shape="sphere", n_points=512, noise_sigma=0.02, seed=0)
    )
    f = extract_features(pts)
    assert f.ndim == 1
    assert f.shape[0] >= 10
    assert np.isfinite(f).all()
