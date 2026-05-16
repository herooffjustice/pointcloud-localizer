import numpy as np
import os
import tempfile
import pytest

from pointcloud_localizer.synthetic import generate_scene, generate_default_mesh, sample_mesh, apply_transform
from pointcloud_localizer.loader import load_pointcloud, save_pointcloud, load_ply, save_ply
from pointcloud_localizer.icp import icp, svd_point_to_point
from pointcloud_localizer.evaluate import rotation_error, translation_error
from pointcloud_localizer.preprocess import voxel_downsample, estimate_normals


class TestICPRecoveryNoiseFree:
    def test_icp_recovers_known_transform(self):
        mesh = generate_default_mesh("box_plane")
        scene = generate_scene(
            mesh=mesh,
            n_points=5000,
            noise_sigma=0.0,
            max_rot_deg=15.0,
            max_trans=0.1,
            seed=42,
        )

        result = icp(
            scene["source"],
            scene["target"],
            max_iterations=100,
            tolerance=1e-8,
            method="point-to-point",
            voxel_size=None,
        )

        rot_err = rotation_error(scene["R_gt"], result["R_est"])
        trans_err = translation_error(scene["t_gt"], result["t_est"])

        assert rot_err < 1.0, f"Rotation error {rot_err:.4f} deg exceeds 1 deg threshold"
        assert trans_err < 0.01, f"Translation error {trans_err:.6f}m exceeds 1cm threshold"

    def test_icp_convergence_decreasing(self):
        mesh = generate_default_mesh("box_plane")
        scene = generate_scene(
            mesh=mesh,
            n_points=3000,
            noise_sigma=0.0,
            max_rot_deg=10.0,
            max_trans=0.05,
            seed=123,
        )

        result = icp(
            scene["source"],
            scene["target"],
            max_iterations=50,
            tolerance=1e-8,
            method="point-to-point",
            voxel_size=None,
        )

        rmse_hist = result["rmse_history"]
        for i in range(1, len(rmse_hist)):
            assert rmse_hist[i] <= rmse_hist[i - 1] + 1e-8, (
                f"RMSE increased at iteration {i}: {rmse_hist[i]:.6f} > {rmse_hist[i-1]:.6f}"
            )

    def test_icp_with_noise(self):
        mesh = generate_default_mesh("box_plane")
        scene = generate_scene(
            mesh=mesh,
            n_points=5000,
            noise_sigma=0.005,
            max_rot_deg=10.0,
            max_trans=0.05,
            seed=42,
        )

        result = icp(
            scene["source"],
            scene["target"],
            max_iterations=100,
            tolerance=1e-6,
            method="point-to-point",
            voxel_size=0.02,
        )

        rot_err = rotation_error(scene["R_gt"], result["R_est"])
        trans_err = translation_error(scene["t_gt"], result["t_est"])

        assert rot_err < 5.0, f"Rotation error {rot_err:.4f} deg too large for noisy case"
        assert trans_err < 0.05, f"Translation error {trans_err:.6f}m too large for noisy case"

    def test_point_to_plane(self):
        mesh = generate_default_mesh("box_plane")
        scene = generate_scene(
            mesh=mesh,
            n_points=5000,
            noise_sigma=0.005,
            max_rot_deg=10.0,
            max_trans=0.05,
            seed=42,
        )

        result = icp(
            scene["source"],
            scene["target"],
            max_iterations=100,
            tolerance=1e-6,
            method="point-to-plane",
            voxel_size=0.02,
        )

        rot_err = rotation_error(scene["R_gt"], result["R_est"])
        trans_err = translation_error(scene["t_gt"], result["t_est"])

        assert rot_err < 10.0, f"PtPl rotation error {rot_err:.4f} deg too large"
        assert trans_err < 0.1, f"PtPl translation error {trans_err:.6f}m too large"

    def test_icp_torus_small_transform(self):
        mesh = generate_default_mesh("torus")
        scene = generate_scene(
            mesh=mesh,
            n_points=5000,
            noise_sigma=0.0,
            max_rot_deg=5.0,
            max_trans=0.02,
            seed=42,
        )

        result = icp(
            scene["source"],
            scene["target"],
            max_iterations=100,
            tolerance=1e-8,
            method="point-to-point",
            voxel_size=None,
        )

        rot_err = rotation_error(scene["R_gt"], result["R_est"])
        trans_err = translation_error(scene["t_gt"], result["t_est"])

        assert rot_err < 1.0, f"Rotation error {rot_err:.4f} deg exceeds 1 deg threshold"
        assert trans_err < 0.01, f"Translation error {trans_err:.6f}m exceeds 1cm threshold"


class TestSVDRegistration:
    def test_svd_exact_correspondences(self):
        R_gt = np.array([
            [0.9962, -0.0872, 0.0],
            [0.0872, 0.9962, 0.0],
            [0.0, 0.0, 1.0],
        ])
        t_gt = np.array([0.1, -0.05, 0.02])

        rng = np.random.default_rng(42)
        source = rng.standard_normal((100, 3))
        target = (source @ R_gt.T) + t_gt

        R_est, t_est = svd_point_to_point(source, target)

        assert np.allclose(R_est, R_gt, atol=1e-6), "SVD rotation mismatch"
        assert np.allclose(t_est, t_gt, atol=1e-6), "SVD translation mismatch"

    def test_svd_identity_transform(self):
        rng = np.random.default_rng(42)
        source = rng.standard_normal((50, 3))

        R_est, t_est = svd_point_to_point(source, source)

        assert np.allclose(R_est, np.eye(3), atol=1e-6), "Identity rotation not recovered"
        assert np.allclose(t_est, np.zeros(3), atol=1e-6), "Identity translation not recovered"


class TestVoxelDownsample:
    def test_reduces_point_count(self):
        rng = np.random.default_rng(42)
        points = rng.standard_normal((10000, 3))
        downsampled, _ = voxel_downsample(points, 0.1)
        assert downsampled.shape[0] < points.shape[0], "Downsampling should reduce point count"
        assert downsampled.shape[1] == 3, "Output should have 3 columns"

    def test_preserves_centroid(self):
        rng = np.random.default_rng(42)
        points = rng.standard_normal((10000, 3))
        centroid_before = np.mean(points, axis=0)
        downsampled, _ = voxel_downsample(points, 0.05)
        centroid_after = np.mean(downsampled, axis=0)
        assert np.allclose(centroid_before, centroid_after, atol=0.05), "Centroid should be approximately preserved"

    def test_zero_voxel_size(self):
        rng = np.random.default_rng(42)
        points = rng.standard_normal((100, 3))
        result, _ = voxel_downsample(points, 0)
        assert result.shape == points.shape, "Zero voxel size should return all points"


class TestEvaluationMetrics:
    def test_rotation_error_zero(self):
        R = np.eye(3)
        assert rotation_error(R, R) < 1e-10, "Identity rotation should have zero error"

    def test_translation_error_zero(self):
        t = np.array([1.0, 2.0, 3.0])
        assert translation_error(t, t) < 1e-10, "Same translation should have zero error"

    def test_rotation_error_known(self):
        angle = 30.0
        rad = np.deg2rad(angle)
        R = np.array([
            [np.cos(rad), -np.sin(rad), 0],
            [np.sin(rad), np.cos(rad), 0],
            [0, 0, 1],
        ])
        err = rotation_error(R, np.eye(3))
        assert abs(err - angle) < 0.1, f"Expected ~{angle} deg, got {err:.4f} deg"

    def test_translation_error_known(self):
        t1 = np.array([0.0, 0.0, 0.0])
        t2 = np.array([3.0, 4.0, 0.0])
        err = translation_error(t1, t2)
        assert abs(err - 5.0) < 1e-10, f"Expected 5.0m, got {err:.4f}m"


class TestLoaderIO:
    def test_ply_roundtrip(self):
        rng = np.random.default_rng(42)
        points = rng.standard_normal((100, 3))
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            path = f.name
        try:
            save_ply(points, path)
            loaded, normals = load_ply(path)
            assert normals is None, "Should have no normals"
            assert loaded.shape == points.shape, f"Shape mismatch: {loaded.shape} vs {points.shape}"
            assert np.allclose(loaded, points, atol=1e-4), "PLY roundtrip failed"
        finally:
            os.unlink(path)

    def test_ply_roundtrip_with_normals(self):
        rng = np.random.default_rng(42)
        points = rng.standard_normal((100, 3))
        normals = rng.standard_normal((100, 3))
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            path = f.name
        try:
            save_ply(points, path, normals=normals)
            loaded_pts, loaded_normals = load_ply(path)
            assert loaded_normals is not None, "Should have normals"
            assert loaded_pts.shape == (100, 3)
            assert np.allclose(loaded_pts, points, atol=1e-4)
            assert np.allclose(loaded_normals, normals, atol=1e-4)
        finally:
            os.unlink(path)

    def test_pcd_roundtrip(self):
        rng = np.random.default_rng(42)
        points = rng.standard_normal((50, 3))
        with tempfile.NamedTemporaryFile(suffix=".pcd", delete=False) as f:
            path = f.name
        try:
            save_pointcloud(points, path)
            loaded = load_pointcloud(path)
            assert loaded.shape == points.shape, f"Shape mismatch: {loaded.shape} vs {points.shape}"
            assert np.allclose(loaded, points, atol=1e-4), "PCD roundtrip failed"
        finally:
            os.unlink(path)

    def test_load_pointcloud_invalid_format(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            load_pointcloud("test.xyz")


class TestNormalEstimation:
    def test_normals_on_plane(self):
        rng = np.random.default_rng(42)
        xy_points = rng.standard_normal((500, 2))
        z_zeros = np.zeros((500, 1))
        plane_pts = np.hstack([xy_points, z_zeros])

        normals = estimate_normals(plane_pts, k_neighbors=30)
        assert normals.shape == (500, 3), f"Wrong shape: {normals.shape}"

        dots = np.abs(np.sum(normals * np.array([0, 0, 1]), axis=1))
        assert np.mean(dots > 0.9) > 0.8, "Most normals on XY plane should point along Z"

    def test_normals_are_unit_vectors(self):
        mesh = generate_default_mesh("box_plane")
        points, _ = sample_mesh(mesh, 1000, seed=42)
        normals = estimate_normals(points, k_neighbors=30)
        norms = np.linalg.norm(normals, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6), "Normals should be unit vectors"


class TestNormalInterpolation:
    def test_barycentric_normal_interpolation(self):
        mesh = generate_default_mesh("box_plane")
        points, normals = sample_mesh(mesh, 5000, seed=42)
        assert normals is not None, "sample_mesh should return normals when mesh has vertex_normals"
        assert normals.shape == (5000, 3)

        norms = np.linalg.norm(normals, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6), "Interpolated normals should be unit vectors"

        unique_normals = np.unique(np.round(normals, decimals=1), axis=0)
        assert unique_normals.shape[0] >= 3, f"Normals should vary across surfaces, got {unique_normals.shape[0]} unique directions"