import numpy as np

from pointcloud_mlops.core.pointcloud_generator import GenerateParams, generate_pointcloud


def test_generate_shapes_have_correct_shape():
    for shape in ["sphere", "cube", "cylinder", "cone", "torus"]:
        pts = generate_pointcloud(
            GenerateParams(shape=shape, n_points=512, noise_sigma=0.02, seed=1)
        )
        assert isinstance(pts, np.ndarray)
        assert pts.shape == (512, 3)
        assert np.isfinite(pts).all()
