import pytest
import numpy as np
from main import project_to_planes, generate_sample_data

def test_projections_return_correct_shapes():
    pointcloud = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    xy_projection, xz_projection, yz_projection = project_to_planes(pointcloud)
    assert xy_projection.shape == (3, 2)
    assert xz_projection.shape == (3, 2)
    assert yz_projection.shape == (3, 2)

def test_projections_handle_empty_pointcloud():
    pointcloud = np.empty((0, 3))
    xy_projection, xz_projection, yz_projection = project_to_planes(pointcloud)
    assert xy_projection.shape == (0, 2)
    assert xz_projection.shape == (0, 2)
    assert yz_projection.shape == (0, 2)

def test_sample_data_has_correct_shape():
    num_points = 100
    pointcloud = generate_sample_data(num_points)
    assert pointcloud.shape == (num_points, 3)

def test_sample_data_handles_zero_points():
    num_points = 0
    pointcloud = generate_sample_data(num_points)
    assert pointcloud.shape == (0, 3)

def test_sample_data_handles_negative_points():
    num_points = -10
    with pytest.raises(ValueError):
        pointcloud = generate_sample_data(num_points)
        assert pointcloud.shape == (0, 3)