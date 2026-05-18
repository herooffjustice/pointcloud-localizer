import json
import os
import csv
import numpy as np

from pointcloud_localizer.loader import load_pointcloud
from pointcloud_localizer.icp import icp
from pointcloud_localizer.synthetic import generate_scene, generate_default_mesh
from pointcloud_localizer.evaluate import evaluate_registration


def downsample_for_web(points, max_points=15000):
    if len(points) <= max_points:
        return points.tolist()
    indices = np.random.choice(len(points), max_points, replace=False)
    return points[indices].tolist()


def main():
    output_dir = "output"
    site_data_dir = os.path.join("site", "data")
    os.makedirs(site_data_dir, exist_ok=True)

    print("Loading point clouds...")
    source = load_pointcloud(os.path.join(output_dir, "source.ply"))
    target = load_pointcloud(os.path.join(output_dir, "target.ply"))

    R_gt = np.load(os.path.join(output_dir, "R_gt.npy"))
    t_gt = np.load(os.path.join(output_dir, "t_gt.npy"))

    print("Running ICP registrations...")
    ptp_result = icp(source, target, method="point-to-point", voxel_size=0.02)
    ptpl_result = icp(source, target, method="point-to-plane", voxel_size=0.02)

    ptp_eval = evaluate_registration(source, target, R_gt, t_gt, ptp_result)
    ptpl_eval = evaluate_registration(source, target, R_gt, t_gt, ptpl_result)

    T_ptp = ptp_result["T_est"]
    T_ptpl = ptpl_result["T_est"]
    source_ptp = (source @ T_ptp[:3, :3].T) + T_ptp[:3, 3]
    source_ptpl = (source @ T_ptpl[:3, :3].T) + T_ptpl[:3, 3]

    print("Exporting point clouds...")
    data = {
        "source": downsample_for_web(source),
        "target": downsample_for_web(target),
        "source_registered_ptp": downsample_for_web(source_ptp),
        "source_registered_ptpl": downsample_for_web(source_ptpl),
        "gt_transform": {
            "R": R_gt.tolist(),
            "t": t_gt.tolist(),
        },
        "ptp_result": {
            "rmse_history": ptp_result["rmse_history"],
            "iterations": ptp_result["iteration_count"],
            "rotation_error_deg": ptp_eval["rotation_error_deg"],
            "translation_error_m": ptp_eval["translation_error_m"],
            "final_rmse": ptp_eval["final_rmse"],
        },
        "ptpl_result": {
            "rmse_history": ptpl_result["rmse_history"],
            "iterations": ptpl_result["iteration_count"],
            "rotation_error_deg": ptpl_eval["rotation_error_deg"],
            "translation_error_m": ptpl_eval["translation_error_m"],
            "final_rmse": ptpl_eval["final_rmse"],
        },
    }

    with open(os.path.join(site_data_dir, "registration.json"), "w") as f:
        json.dump(data, f)

    print("Exporting sweep results...")
    for sweep_dir, sweep_name in [
        (output_dir, "torus"),
        (os.path.join(output_dir, "box_plane_sweep"), "box_plane"),
    ]:
        csv_path = os.path.join(sweep_dir, "sweep_results.csv")
        if not os.path.exists(csv_path):
            continue
        rows = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "noise_sigma": float(row["noise_sigma"]),
                    "max_rotation_deg": float(row["max_rotation_deg"]),
                    "max_translation_m": float(row["max_translation_m"]),
                    "rotation_error_deg": float(row["rotation_error_deg"]),
                    "translation_error_m": float(row["translation_error_m"]),
                    "final_rmse": float(row["final_rmse"]),
                    "iterations": int(row["iterations"]),
                })

        noise_levels = sorted(set(r["noise_sigma"] for r in rows))
        misalign_levels = sorted(set(
            (r["max_rotation_deg"], r["max_translation_m"]) for r in rows
        ), key=lambda x: x[0])

        rot_grid = []
        trans_grid = []
        rmse_grid = []
        for noise in noise_levels:
            rot_row = []
            trans_row = []
            rmse_row = []
            for max_rot, max_trans in misalign_levels:
                match = [r for r in rows
                         if abs(r["noise_sigma"] - noise) < 1e-9
                         and abs(r["max_rotation_deg"] - max_rot) < 1e-9
                         and abs(r["max_translation_m"] - max_trans) < 1e-9]
                if match:
                    m = match[0]
                    rot_row.append(m["rotation_error_deg"])
                    trans_row.append(m["translation_error_m"])
                    rmse_row.append(m["final_rmse"])
                else:
                    rot_row.append(0)
                    trans_row.append(0)
                    rmse_row.append(0)
            rot_grid.append(rot_row)
            trans_grid.append(trans_row)
            rmse_grid.append(rmse_row)

        sweep_data = {
            "mesh_type": sweep_name,
            "noise_levels": noise_levels,
            "misalignment_labels": [f"{mr}deg / {mt}m" for mr, mt in misalign_levels],
            "rotation_errors": rot_grid,
            "translation_errors": trans_grid,
            "rmse_values": rmse_grid,
            "rows": rows,
        }
        with open(os.path.join(site_data_dir, f"sweep_{sweep_name}.json"), "w") as f:
            json.dump(sweep_data, f)

    print("Generating algorithm step data...")
    mesh = generate_default_mesh("torus")
    scene = generate_scene(mesh, n_points=500, noise_sigma=0.005, max_rot_deg=15, max_trans=0.1, seed=99)
    src = scene["source"]
    tgt = scene["target"]

    steps = []
    current = src.copy()
    from scipy.spatial import KDTree
    tree = KDTree(tgt)

    for i in range(8):
        dists, indices = tree.query(current)
        rmse = np.sqrt(np.mean(dists ** 2))
        steps.append({
            "iteration": i,
            "source": downsample_for_web(current, max_points=2000),
            "rmse": rmse,
        })
        matched_src = current
        matched_tgt = tgt[indices]
        from pointcloud_localizer.icp import svd_point_to_point
        R_step, t_step = svd_point_to_point(matched_src, matched_tgt)
        current = (current @ R_step.T) + t_step

    with open(os.path.join(site_data_dir, "icp_steps.json"), "w") as f:
        json.dump({"target": downsample_for_web(tgt, max_points=2000), "steps": steps}, f)

    print("All data exported to site/data/")


if __name__ == "__main__":
    main()
