import numpy as np
import os

from pointcloud_localizer.icp import icp
from pointcloud_localizer.synthetic import generate_scene, generate_default_mesh, sample_mesh, add_noise, apply_transform, random_transform
from pointcloud_localizer.loader import load_mesh


def rotation_error(R_gt, R_est):
    R_err = R_gt.T @ R_est
    trace = np.trace(R_err)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def translation_error(t_gt, t_est):
    return np.linalg.norm(t_est - t_gt)


def evaluate_registration(source, target, R_gt, t_gt, result, output_dir=None, label=""):
    R_est = result["R_est"]
    t_est = result["t_est"]

    rot_err = rotation_error(R_gt, R_est)
    trans_err = translation_error(t_gt, t_est)

    eval_result = {
        "rotation_error_deg": rot_err,
        "translation_error_m": trans_err,
        "final_rmse": result["rmse_history"][-1] if result["rmse_history"] else float("inf"),
        "iterations": result["iteration_count"],
        "rmse_history": result["rmse_history"],
        "method": result["method"],
    }

    if output_dir is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        T_est = result["T_est"]
        transformed = (source @ T_est[:3, :3].T) + T_est[:3, 3]

        fig = plt.figure(figsize=(18, 5))

        ax1 = fig.add_subplot(131, projection="3d")
        ax1.scatter(source[:, 0], source[:, 1], source[:, 2], c="blue", s=1, alpha=0.5, label="Source")
        ax1.scatter(target[:, 0], target[:, 1], target[:, 2], c="red", s=1, alpha=0.5, label="Target")
        ax1.set_title("Before Registration")
        ax1.legend()

        ax2 = fig.add_subplot(132, projection="3d")
        ax2.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], c="blue", s=1, alpha=0.5, label="Transformed Source")
        ax2.scatter(target[:, 0], target[:, 1], target[:, 2], c="red", s=1, alpha=0.5, label="Target")
        ax2.set_title("After Registration")
        ax2.legend()

        ax3 = fig.add_subplot(133)
        ax3.plot(range(1, len(result["rmse_history"]) + 1), result["rmse_history"], "o-")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("RMSE")
        ax3.set_title(f"RMSE Convergence ({label})")
        ax3.grid(True)

        plt.tight_layout()
        tag = f"_{label}" if label else ""
        plt.savefig(os.path.join(output_dir, f"registration{tag}.png"), dpi=150)
        plt.close()

    return eval_result


def plot_rmse_convergence(rmse_history, save_path, label=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(rmse_history) + 1), rmse_history, "o-")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title(f"RMSE Convergence Curve{f' ({label})' if label else ''}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_before_registration(source, target, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(source[:, 0], source[:, 1], source[:, 2], c="blue", s=1, alpha=0.5, label="Source")
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], c="red", s=1, alpha=0.5, label="Target")
    ax.set_title("Before Registration")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_after_registration(source_transformed, target, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(source_transformed[:, 0], source_transformed[:, 1], source_transformed[:, 2], c="blue", s=1, alpha=0.5, label="Transformed Source")
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], c="red", s=1, alpha=0.5, label="Target")
    ax.set_title("After Registration")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def robustness_sweep(
    mesh=None,
    mesh_type="torus",
    noise_levels=(0, 0.005, 0.02),
    misalignment_levels=((5, 0.02), (30, 0.1), (60, 0.3)),
    n_points=10000,
    voxel_size=0.02,
    max_iterations=50,
    tolerance=1e-6,
    method="point-to-point",
    output_dir=None,
    seed=42,
):
    if mesh is None:
        mesh = generate_default_mesh(mesh_type)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    results = []

    for i, noise_sigma in enumerate(noise_levels):
        for j, misalignment in enumerate(misalignment_levels):
            max_rot, max_trans = misalignment

            sweep_seed = seed + i * len(misalignment_levels) + j
            source, _ = sample_mesh(mesh, n_points, seed=sweep_seed * 3)
            R_gt, t_gt = random_transform(max_rot, max_trans, seed=sweep_seed * 3 + 1)
            target_clean = apply_transform(source, R_gt, t_gt)
            target, _ = add_noise(target_clean, sigma=noise_sigma, seed=sweep_seed * 3 + 2)

            result = icp(
                source,
                target,
                init_T=None,
                max_iterations=max_iterations,
                tolerance=tolerance,
                method=method,
                voxel_size=voxel_size,
            )

            eval_result = evaluate_registration(source, target, R_gt, t_gt, result)

            results.append({
                "noise_sigma": noise_sigma,
                "max_rotation_deg": max_rot,
                "max_translation_m": max_trans,
                "rotation_error_deg": eval_result["rotation_error_deg"],
                "translation_error_m": eval_result["translation_error_m"],
                "final_rmse": eval_result["final_rmse"],
                "iterations": eval_result["iterations"],
            })

    if output_dir is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_noise = len(noise_levels)
        n_misalign = len(misalignment_levels)

        rot_errors = np.array([r["rotation_error_deg"] for r in results]).reshape(n_noise, n_misalign)
        trans_errors = np.array([r["translation_error_m"] for r in results]).reshape(n_noise, n_misalign)

        misalign_labels = [f"{m[0]}deg/{m[1]}m" for m in misalignment_levels]
        noise_labels = [f"σ={n}" for n in noise_levels]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        im1 = axes[0].imshow(rot_errors, cmap="YlOrRd", aspect="auto")
        axes[0].set_xticks(range(n_misalign))
        axes[0].set_xticklabels(misalign_labels)
        axes[0].set_yticks(range(n_noise))
        axes[0].set_yticklabels(noise_labels)
        axes[0].set_xlabel("Misalignment")
        axes[0].set_ylabel("Noise Level")
        axes[0].set_title("Rotation Error (degrees)")
        for i in range(n_noise):
            for j in range(n_misalign):
                axes[0].text(j, i, f"{rot_errors[i, j]:.2f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(trans_errors, cmap="YlOrRd", aspect="auto")
        axes[1].set_xticks(range(n_misalign))
        axes[1].set_xticklabels(misalign_labels)
        axes[1].set_yticks(range(n_noise))
        axes[1].set_yticklabels(noise_labels)
        axes[1].set_xlabel("Misalignment")
        axes[1].set_ylabel("Noise Level")
        axes[1].set_title("Translation Error (meters)")
        for i in range(n_noise):
            for j in range(n_misalign):
                axes[1].text(j, i, f"{trans_errors[i, j]:.4f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im2, ax=axes[1])

        plt.suptitle("Robustness Sweep: Registration Quality vs Noise & Misalignment")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sweep_heatmaps.png"), dpi=150)
        plt.close()

        csv_path = os.path.join(output_dir, "sweep_results.csv")
        with open(csv_path, "w") as f:
            f.write("noise_sigma,max_rotation_deg,max_translation_m,rotation_error_deg,translation_error_m,final_rmse,iterations\n")
            for r in results:
                f.write(f"{r['noise_sigma']},{r['max_rotation_deg']},{r['max_translation_m']},"
                        f"{r['rotation_error_deg']:.6f},{r['translation_error_m']:.6f},"
                        f"{r['final_rmse']:.6f},{r['iterations']}\n")

        fig2, ax_table = plt.subplots(figsize=(12, 4))
        ax_table.axis("off")
        cell_text = []
        for r in results:
            cell_text.append([
                f"{r['noise_sigma']}",
                f"{r['max_rotation_deg']}deg / {r['max_translation_m']}m",
                f"{r['rotation_error_deg']:.3f}",
                f"{r['translation_error_m']:.5f}",
                f"{r['final_rmse']:.6f}",
                f"{r['iterations']}",
            ])
        col_labels = ["Noise σ (m)", "Misalignment", "Rot Error (°)", "Trans Error (m)", "Final RMSE", "Iterations"]
        table = ax_table.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax_table.set_title("Robustness Sweep Results", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sweep_summary.png"), dpi=150)
        plt.close()

    return results