import numpy as np
from scipy.spatial import KDTree


def voxel_downsample(points, voxel_size):
    if voxel_size <= 0:
        return points.copy(), np.arange(len(points))

    points = np.asarray(points, dtype=np.float64)
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)

    _, unique_inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    n_voxels = unique_inverse.max() + 1

    summed = np.zeros((n_voxels, 3), dtype=np.float64)
    counts = np.zeros(n_voxels, dtype=np.float64)

    np.add.at(summed, unique_inverse, points)
    np.add.at(counts, unique_inverse, 1.0)

    downsampled = summed / counts[:, np.newaxis]

    representative_indices = np.zeros(n_voxels, dtype=np.int64)
    representative_indices[unique_inverse] = np.arange(len(points))

    return downsampled, representative_indices


def estimate_normals(points, k_neighbors=30):
    pcd = points.astype(np.float64)
    n_pts = pcd.shape[0]

    if n_pts < k_neighbors:
        k_neighbors = n_pts

    tree = KDTree(pcd)
    _, indices = tree.query(pcd, k=k_neighbors)

    neighbor_pts = pcd[indices]  # (n_pts, k, 3)
    centroids = neighbor_pts.mean(axis=1, keepdims=True)  # (n_pts, 1, 3)
    centered = neighbor_pts - centroids  # (n_pts, k, 3)

    centered_T = centered.transpose(0, 2, 1)  # (n_pts, 3, k)
    cov_matrices = centered_T @ centered / k_neighbors  # (n_pts, 3, 3)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices)  # (n_pts, 3), (n_pts, 3, 3)
    normals = eigenvectors[:, :, 0]  # (n_pts, 3) — smallest eigenvalue's eigenvector

    centroid_all = pcd.mean(axis=0)  # (3,)
    orientations = np.sum(normals * (pcd - centroid_all), axis=1)  # (n_pts,)
    normals[orientations < 0] *= -1

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normals = normals / norms

    return normals


def preprocess_for_icp(points, voxel_size=None, compute_normals=False, k_neighbors=30):
    result = {"points": points.copy()}
    representative_indices = np.arange(len(points))

    if voxel_size is not None and voxel_size > 0:
        result["points"], representative_indices = voxel_downsample(result["points"], voxel_size)

    if compute_normals:
        result["normals"] = estimate_normals(result["points"], k_neighbors)

    result["representative_indices"] = representative_indices
    return result