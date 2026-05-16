import numpy as np
from scipy.spatial import KDTree

from pointcloud_localizer.preprocess import voxel_downsample, estimate_normals


def find_correspondences(source, target_kdtree):
    distances, indices = target_kdtree.query(source)
    return distances, indices


def compute_rmse(distances):
    return np.sqrt(np.mean(distances ** 2))


def svd_point_to_point(source_matched, target_matched):
    centroid_source = np.mean(source_matched, axis=0)
    centroid_target = np.mean(target_matched, axis=0)

    q_s = source_matched - centroid_source
    q_t = target_matched - centroid_target

    H = q_s.T @ q_t

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_target - R @ centroid_source
    return R, t


def solve_point_to_plane(source_matched, target_matched, target_normals_matched):
    n_pts = source_matched.shape[0]
    A = np.zeros((n_pts, 6))
    b = np.zeros(n_pts)

    for i in range(n_pts):
        p_s = source_matched[i]
        p_t = target_matched[i]
        n = target_normals_matched[i]

        A[i, :3] = np.cross(p_s, n)
        A[i, 3:] = n
        b[i] = n @ (p_t - p_s)

    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    alpha, beta, gamma = result[0], result[1], result[2]
    tx, ty, tz = result[3], result[4], result[5]

    angle = np.sqrt(alpha**2 + beta**2 + gamma**2)

    if angle < 1e-10:
        R = np.eye(3)
    else:
        axis = np.array([alpha, beta, gamma]) / angle
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        R = np.eye(3) + sin_angle * K + (1 - cos_angle) * (K @ K)
        U_r, S_r, Vt_r = np.linalg.svd(R)
        R = Vt_r.T @ U_r.T
        if np.linalg.det(R) < 0:
            Vt_r[-1, :] *= -1
            R = Vt_r.T @ U_r.T

    t = np.array([tx, ty, tz])
    return R, t


def _interpolate_transform(R_full, t_full, alpha):
    R_interp = alpha * R_full + (1 - alpha) * np.eye(3)
    U, S, Vt = np.linalg.svd(R_interp)
    R_interp = Vt.T @ U.T
    if np.linalg.det(R_interp) < 0:
        Vt[-1, :] *= -1
        R_interp = Vt.T @ U.T
    t_interp = alpha * t_full
    return R_interp, t_interp


def icp(
    source,
    target,
    init_T=None,
    max_iterations=50,
    tolerance=1e-6,
    method="point-to-point",
    target_normals=None,
    voxel_size=None,
):
    source = np.asarray(source, dtype=np.float64).copy()
    target = np.asarray(target, dtype=np.float64).copy()

    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Source must be (N, 3), got {source.shape}")
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError(f"Target must be (M, 3), got {target.shape}")
    if source.shape[0] < 3:
        raise ValueError(f"Source needs at least 3 points, got {source.shape[0]}")
    if target.shape[0] < 3:
        raise ValueError(f"Target needs at least 3 points, got {target.shape[0]}")
    if method not in ("point-to-point", "point-to-plane"):
        raise ValueError(f"Unknown method: {method}. Use 'point-to-point' or 'point-to-plane'")

    if voxel_size is not None and voxel_size > 0:
        source, _ = voxel_downsample(source, voxel_size)
        target, _ = voxel_downsample(target, voxel_size)
        if target_normals is not None and method == "point-to-plane":
            target_normals = estimate_normals(target)

    if init_T is not None:
        R_init, t_init = init_T[:3, :3], init_T[:3, 3]
        source = (source @ R_init.T) + t_init

    if method == "point-to-plane" and target_normals is None:
        target_normals = estimate_normals(target)

    T_accumulated = np.eye(4)
    if init_T is not None:
        T_accumulated = init_T.copy()

    rmse_history = []
    target_kdtree = KDTree(target)

    prev_rmse = np.inf

    for iteration in range(max_iterations):
        distances, indices = find_correspondences(source, target_kdtree)
        current_rmse = compute_rmse(distances)
        rmse_history.append(current_rmse)

        if iteration > 0 and abs(prev_rmse - current_rmse) < tolerance:
            break

        src_matched = source
        tgt_matched = target[indices]

        if method == "point-to-point":
            R_step, t_step = svd_point_to_point(src_matched, tgt_matched)
            source = (source @ R_step.T) + t_step
        elif method == "point-to-plane":
            normals_matched = target_normals[indices]
            R_step, t_step = solve_point_to_plane(
                src_matched, tgt_matched, normals_matched
            )

            source_new = (source @ R_step.T) + t_step
            dist_new, _ = find_correspondences(source_new, target_kdtree)
            rmse_new = compute_rmse(dist_new)

            step_applied = True
            if rmse_new <= current_rmse:
                source = source_new
            else:
                best_alpha = None
                best_rmse = current_rmse
                for trial_alpha in [0.5, 0.25, 0.125, 0.0625]:
                    R_trial, t_trial = _interpolate_transform(R_step, t_step, trial_alpha)
                    source_trial = (source @ R_trial.T) + t_trial
                    dist_trial, _ = find_correspondences(source_trial, target_kdtree)
                    rmse_trial = compute_rmse(dist_trial)
                    if rmse_trial < best_rmse:
                        best_rmse = rmse_trial
                        best_alpha = trial_alpha

                if best_alpha is not None:
                    R_step, t_step = _interpolate_transform(R_step, t_step, best_alpha)
                    source = (source @ R_step.T) + t_step
                else:
                    step_applied = False

            if not step_applied:
                R_step = np.eye(3)
                t_step = np.zeros(3)
        else:
            raise ValueError(f"Unknown ICP method: {method}")

        T_step = np.eye(4)
        T_step[:3, :3] = R_step
        T_step[:3, 3] = t_step
        T_accumulated = T_step @ T_accumulated

        prev_rmse = rmse_history[-1]

    R_est = T_accumulated[:3, :3]
    t_est = T_accumulated[:3, 3]

    return {
        "T_est": T_accumulated,
        "R_est": R_est,
        "t_est": t_est,
        "rmse_history": rmse_history,
        "iteration_count": len(rmse_history),
        "method": method,
    }