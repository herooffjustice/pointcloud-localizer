import numpy as np

from pointcloud_localizer.synthetic import generate_scene, generate_default_mesh, sample_mesh, add_noise, apply_transform, random_transform
from pointcloud_localizer.loader import load_pointcloud, load_mesh, save_pointcloud
from pointcloud_localizer.preprocess import voxel_downsample, estimate_normals
from pointcloud_localizer.icp import icp
from pointcloud_localizer.evaluate import (
    evaluate_registration,
    robustness_sweep,
    plot_rmse_convergence,
    plot_before_registration,
    plot_after_registration,
    rotation_error,
    translation_error,
)


def cmd_generate(args):
    if args.mesh:
        mesh = load_mesh(args.mesh)
    else:
        mesh = generate_default_mesh(args.mesh_type)

    import os
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    result = generate_scene(
        mesh=mesh,
        n_points=args.n_points,
        noise_sigma=args.noise,
        max_rot_deg=args.rotation,
        max_trans=args.translation,
        seed=args.seed,
        source_path=os.path.join(output_dir, "source.ply"),
        target_path=os.path.join(output_dir, "target.ply"),
    )

    np.save(os.path.join(output_dir, "R_gt.npy"), result["R_gt"])
    np.save(os.path.join(output_dir, "t_gt.npy"), result["t_gt"])

    print(f"Generated scene with {args.n_points} points")
    print(f"  Noise: sigma={args.noise}m, Rotation: up to {args.rotation} deg, Translation: up to {args.translation}m")
    print(f"  Files saved to {output_dir}/")


def cmd_register(args):
    import os

    source = load_pointcloud(args.source)
    target = load_pointcloud(args.target)

    has_gt = args.R_gt is not None and args.t_gt is not None
    if has_gt:
        R_gt = np.load(args.R_gt)
        t_gt = np.load(args.t_gt)
    else:
        R_gt = np.eye(3)
        t_gt = np.zeros(3)
        print("Warning: No ground truth provided. Skipping evaluation metrics.")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    before_dir = os.path.join(output_dir, "before")
    after_dir = os.path.join(output_dir, "after")
    os.makedirs(before_dir, exist_ok=True)
    os.makedirs(after_dir, exist_ok=True)

    plot_before_registration(source, target, os.path.join(before_dir, "before_registration.png"))

    result = icp(
        source,
        target,
        init_T=None,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        method=args.method,
        voxel_size=args.voxel_size,
    )

    T_est = result["T_est"]
    transformed = (source @ T_est[:3, :3].T) + T_est[:3, 3]

    label = args.method
    plot_after_registration(transformed, target, os.path.join(after_dir, f"after_registration_{label}.png"))

    plot_rmse_convergence(
        result["rmse_history"],
        os.path.join(output_dir, f"rmse_convergence_{label}.png"),
        label=label,
    )

    if has_gt:
        eval_result = evaluate_registration(source, target, R_gt, t_gt, result, output_dir=output_dir, label=label)
        print(f"\nRegistration Results ({args.method}):")
        print(f"  Iterations: {eval_result['iterations']}")
        print(f"  Final RMSE: {eval_result['final_rmse']:.6f}")
        print(f"  Rotation Error: {eval_result['rotation_error_deg']:.4f} deg")
        print(f"  Translation Error: {eval_result['translation_error_m']:.6f}m")
    else:
        print(f"\nRegistration Results ({args.method}):")
        print(f"  Iterations: {result['iteration_count']}")
        print(f"  Final RMSE: {result['rmse_history'][-1]:.6f}")

    print(f"\nPlots saved to {output_dir}/")


def cmd_sweep(args):
    import os

    if args.mesh:
        mesh = load_mesh(args.mesh)
    else:
        mesh = generate_default_mesh(args.mesh_type)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    noise_levels = [0, 0.005, 0.02]
    misalignment_levels = [(5, 0.02), (30, 0.1), (60, 0.3)]

    results = robustness_sweep(
        mesh=mesh,
        noise_levels=noise_levels,
        misalignment_levels=misalignment_levels,
        n_points=args.n_points,
        voxel_size=args.voxel_size,
        max_iterations=args.max_iterations,
        method=args.method,
        output_dir=output_dir,
        seed=args.seed,
    )

    print(f"\nRobustness Sweep Results:")
    print(f"{'Noise sigma':>10} {'Misalignment':>15} {'Rot Error (deg)':>16} {'Trans Error (m)':>16} {'Final RMSE':>12} {'Iters':>6}")
    print("-" * 78)
    for r in results:
        mis = f"{r['max_rotation_deg']}deg/{r['max_translation_m']}m"
        print(f"{r['noise_sigma']:>10.3f} {mis:>15} {r['rotation_error_deg']:>16.4f} {r['translation_error_m']:>16.6f} {r['final_rmse']:>12.6f} {r['iterations']:>6}")

    print(f"\nResults saved to {output_dir}/")


def cmd_test(args):
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_icp.py", "-v"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    sys.exit(result.returncode)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="pointcloud-localizer: ICP-based point cloud registration"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    gen_parser = subparsers.add_parser("generate", help="Generate a synthetic scene")
    gen_parser.add_argument("--mesh", type=str, default=None, help="Path to mesh file (OBJ/PLY/STL)")
    gen_parser.add_argument("--mesh-type", type=str, default="torus", choices=["torus", "box_plane", "sphere"], help="Built-in mesh type")
    gen_parser.add_argument("--n-points", type=int, default=10000, help="Number of points to sample")
    gen_parser.add_argument("--noise", type=float, default=0.0, help="Gaussian noise sigma (meters)")
    gen_parser.add_argument("--rotation", type=float, default=30.0, help="Max rotation (degrees)")
    gen_parser.add_argument("--translation", type=float, default=0.1, help="Max translation (meters)")
    gen_parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    reg_parser = subparsers.add_parser("register", help="Register two point clouds")
    reg_parser.add_argument("--source", type=str, required=True, help="Path to source point cloud")
    reg_parser.add_argument("--target", type=str, required=True, help="Path to target point cloud")
    reg_parser.add_argument("--R-gt", type=str, default=None, help="Path to ground-truth rotation (.npy)")
    reg_parser.add_argument("--t-gt", type=str, default=None, help="Path to ground-truth translation (.npy)")
    reg_parser.add_argument("--method", type=str, default="point-to-point", choices=["point-to-point", "point-to-plane"], help="ICP method")
    reg_parser.add_argument("--max-iterations", type=int, default=50, help="Max ICP iterations")
    reg_parser.add_argument("--tolerance", type=float, default=1e-6, help="Convergence tolerance")
    reg_parser.add_argument("--voxel-size", type=float, default=0.02, help="Voxel size for downsampling (0 to skip)")
    reg_parser.add_argument("--output-dir", type=str, default="output", help="Output directory")

    sweep_parser = subparsers.add_parser("sweep", help="Run robustness sweep")
    sweep_parser.add_argument("--mesh", type=str, default=None, help="Path to mesh file")
    sweep_parser.add_argument("--mesh-type", type=str, default="torus", choices=["torus", "box_plane", "sphere"], help="Built-in mesh type")
    sweep_parser.add_argument("--n-points", type=int, default=10000, help="Number of points to sample")
    sweep_parser.add_argument("--method", type=str, default="point-to-point", choices=["point-to-point", "point-to-plane"], help="ICP method")
    sweep_parser.add_argument("--voxel-size", type=float, default=0.02, help="Voxel size for downsampling")
    sweep_parser.add_argument("--max-iterations", type=int, default=50, help="Max ICP iterations")
    sweep_parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    sweep_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    test_parser = subparsers.add_parser("test", help="Run verification tests")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "register":
        cmd_register(args)
    elif args.command == "sweep":
        cmd_sweep(args)
    elif args.command == "test":
        cmd_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()