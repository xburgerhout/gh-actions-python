import numpy as np
import argparse
import matplotlib.pyplot as plt
from typing import Tuple

def project_to_planes(pointcloud: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projects the points to the X, Y, and Z planes.

    Parameters:
    pointcloud (numpy.ndarray): A (N, 3) array representing the point cloud.

    Returns:
    tuple: Three numpy arrays representing the projections on the X, Y, and Z planes.
    """
    xy_projection = pointcloud[:, :2]
    xz_projection = pointcloud[:, [0, 2]]
    yz_projection = pointcloud[:, 1:]

    return xy_projection, xz_projection, yz_projection

def generate_sample_data(num_points: int=100) -> np.ndarray:
    """
    Generates sample input data for the point cloud using three different distributions.

    Parameters:
    num_points (int): Number of points to generate for each coordinate.

    Returns:
    numpy.ndarray: A (num_points, 3) array representing the point cloud.
    """
    x = np.random.normal(loc=0.0, scale=1.0, size=num_points)
    y = np.random.uniform(low=-1.0, high=1.0, size=num_points)
    z = np.random.exponential(scale=1.0, size=num_points)

    pointcloud = np.vstack((x, y, z)).T
    return pointcloud

def plot_projections(pointcloud: np.ndarray,
                     xy_projection: np.ndarray,
                     xz_projection: np.ndarray,
                     yz_projection: np.ndarray):
    """
    Plots the input data and the three projection results.

    Parameters:
    pointcloud (numpy.ndarray): The original point cloud.
    xy_projection (numpy.ndarray): The projection on the XY plane.
    xz_projection (numpy.ndarray): The projection on the XZ plane.
    yz_projection (numpy.ndarray): The projection on the YZ plane.
    """
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(221, projection="3d")
    ax1.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], c="r", marker="o")
    ax1.set_title("Original Point Cloud")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    #ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(222)
    ax2.scatter(xy_projection[:, 0], xy_projection[:, 1], c="g", marker="o")
    ax2.set_title("XY Projection")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    ax3 = fig.add_subplot(223)
    ax3.scatter(xz_projection[:, 0], xz_projection[:, 1], c="b", marker="o")
    ax3.set_title("XZ Projection")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")

    ax4 = fig.add_subplot(224)
    ax4.scatter(yz_projection[:, 0], yz_projection[:, 1], c="m", marker="o")
    ax4.set_title("YZ Projection")
    ax4.set_xlabel("Y")
    ax4.set_ylabel("Z")

    plt.tight_layout()
    plt.show()

def main() -> None:
    parser = argparse.ArgumentParser(description="Project point cloud to X, Y, and Z planes.")
    parser.add_argument("--num_points", type=int, default=100, help="Number of points to generate for each coordinate.")

    args = parser.parse_args()
    num_points = args.num_points

    pointcloud = generate_sample_data(num_points)
    xy_projection, xz_projection, yz_projection = project_to_planes(pointcloud)
    plot_projections(pointcloud, xy_projection, xz_projection, yz_projection)

if __name__ == "__main__":
    main()