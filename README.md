# pointcloud-localizer

A tool for point cloud registration using ICP (Iterative Closest Point). Takes two overlapping point clouds, estimates the rigid transform (rotation + translation) between them, and evaluates registration accuracy.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

**Dependencies:** numpy, scipy, matplotlib, pytest

```bash
# Verify installation
pytest tests/test_icp.py -v
```

## How to Run

### Generate a synthetic scene

```bash
python -m pointcloud_localizer generate \
    --mesh-type box_plane \
    --n-points 10000 \
    --noise 0.005 \
    --rotation 15.0 \
    --translation 0.1 \
    --output-dir output \
    --seed 42
```

### Register two point clouds

```bash
# Point-to-Point (default, more robust)
python -m pointcloud_localizer register \
    --source output/source.ply \
    --target output/target.ply \
    --R-gt output/R_gt.npy \
    --t-gt output/t_gt.npy \
    --method point-to-point \
    --voxel-size 0.02 \
    --output-dir output

# Point-to-Plane (faster convergence, needs good normals)
python -m pointcloud_localizer register \
    --source output/source.ply \
    --target output/target.ply \
    --R-gt output/R_gt.npy \
    --t-gt output/t_gt.npy \
    --method point-to-plane \
    --voxel-size 0.02 \
    --output-dir output
```

### Run robustness sweep

```bash
python -m pointcloud_localizer sweep --mesh-type torus --output-dir output --seed 42
python -m pointcloud_localizer sweep --mesh-type box_plane --output-dir output/box_plane_sweep --seed 42
```

### Run tests

```bash
python -m pointcloud_localizer test
# or:
pytest tests/test_icp.py -v
```

### Reproducing all outputs

```bash
python -m pointcloud_localizer generate --mesh-type box_plane --n-points 10000 --noise 0.005 --rotation 15 --translation 0.1 --output-dir output --seed 42
python -m pointcloud_localizer register --source output/source.ply --target output/target.ply --R-gt output/R_gt.npy --t-gt output/t_gt.npy --method point-to-point --voxel-size 0.02 --output-dir output
python -m pointcloud_localizer register --source output/source.ply --target output/target.ply --R-gt output/R_gt.npy --t-gt output/t_gt.npy --method point-to-plane --voxel-size 0.02 --output-dir output
python -m pointcloud_localizer sweep --mesh-type torus --output-dir output --seed 42
python -m pointcloud_localizer sweep --mesh-type box_plane --output-dir output/box_plane_sweep --seed 42
```

## Project Structure

```
pointcloud-localizer/
├── pointcloud_localizer/
│   ├── __init__.py          # Package exports
│   ├── __main__.py          # python -m support
│   ├── loader.py            # PLY/PCD I/O, mesh loading (OBJ/PLY/STL), transform utilities
│   ├── synthetic.py         # Scene generation: mesh sampling, transforms, noise
│   ├── preprocess.py        # Voxel downsampling, PCA-based normal estimation
│   ├── icp.py               # ICP registration (PtP via SVD + PtPl with line search)
│   ├── evaluate.py          # Error metrics, plotting, robustness sweep
│   └── cli.py               # CLI entry point (generate, register, sweep, test)
├── tests/
│   └── test_icp.py          # 21 tests
├── output/                  # Generated results
│   ├── before/              # Before-registration visualization
│   ├── after/               # After-registration (PtP and PtPl)
│   ├── registration_*.png   # Combined before/after/RMSE figures
│   ├── rmse_convergence_*.png
│   ├── source.ply, target.ply
│   ├── R_gt.npy, t_gt.npy
│   ├── sweep_heatmaps.png
│   ├── sweep_results.csv
│   ├── sweep_summary.png
│   └── box_plane_sweep/
├── personal_notes/          # Handwritten learning notes
├── README.md
├── PROMPTS.md
├── requirements.txt
└── setup.py
```

## Design Notes

### ICP Implementation

**Point-to-Point (default):** SVD-based closed-form solution for the optimal rigid transform given correspondences. Center both point sets, form cross-covariance H, decompose with SVD to get R = V@U^T, then t = centroid_target - R@centroid_source. A reflection check (det(R) < 0) ensures a proper rotation. Chosen as default because it doesn't require normals, provably reduces RMSE each iteration, and is globally optimal for fixed correspondences.

**Point-to-Plane (optional):** Linearizes rotation as R ≈ I + [ω]× and solves a 6-variable least-squares system per iteration. Faster convergence on well-conditioned geometry, but can overshoot on flat or symmetric surfaces. Includes backtracking line search: try full step, then α = 0.5, 0.25, 0.125, 0.0625. If no step improves RMSE, skip the iteration.

### Voxel Downsampling

Implemented from scratch using numpy binning. Points are assigned to 3D voxel cells via `np.floor(points / voxel_size)`, then averaged per cell. Reduces point count while preserving shape, making ICP feasible on large clouds.

### Normal Estimation

PCA-based: for each point, find k nearest neighbors via scipy KDTree, compute the covariance matrix, take the eigenvector with smallest eigenvalue as the surface normal. Orient outward by flipping normals pointing toward the cloud centroid. Batched using `np.linalg.eigh` instead of per-point loops.

### Mesh Generation

Three built-in meshes:
- **torus** — smooth surface with rotational symmetry; demonstrates ICP's local-minimum problem
- **box_plane** — asymmetric (box on a flat plane); ICP converges reliably, used for verification tests
- **sphere** — highly symmetric; demonstrates PtPl limitations

Surface sampling uses area-weighted triangle selection and barycentric interpolation for both positions and normals.

### Robustness Sweep

3 noise levels (σ = 0, 0.005, 0.02 m) × 3 misalignment magnitudes (5°/0.02m, 30°/0.1m, 60°/0.3m), with independent random seeds per condition.

Key findings:
- **Low noise, small misalignment** → near-perfect recovery
- **Higher noise** → graceful degradation
- **Larger misalignment on symmetric geometry** → ICP converges to local minima (torus: 13-45° rotation error at 30-60° misalignment)
- **Larger misalignment on asymmetric geometry** → ICP still works reasonably (box_plane: 0.004-4.6° at 30° misalignment)

### Notable Bugs Found During Development

1. **Random transform generation:** `scipy.spatial.transform.Rotation.random()` generates rotations up to 180° regardless of a max_angle parameter. Fixed by sampling angle from `Uniform(0, max_rot_deg)` and axis from normalized `standard_normal(3)`, then using `Rotation.from_rotvec(axis * angle)`.

2. **Barycentric normal interpolation:** The position formula assigns barycentric coordinates (w, u, v) to vertices (v0, v1, v2), but the original code applied (u, v, w) — a cyclic shift. This produced incorrect surface normals that degraded PtPl accuracy. Fixed by reordering to w·n0 + u·n1 + v·n2.

## Test Suite

21 tests across 6 classes:

| Class | Tests | What it verifies |
|-------|-------|-------------------|
| TestICPRecoveryNoiseFree | 5 | <1°/<1cm recovery, convergence, noise, PtPl, torus |
| TestSVDRegistration | 2 | Exact correspondences, identity transform |
| TestVoxelDownsample | 3 | Count reduction, centroid preservation, zero-size |
| TestEvaluationMetrics | 4 | Zero errors, known rotation, known distance |
| TestLoaderIO | 4 | PLY/PCD round-trip, invalid format |
| TestNormalEstimation | 3 | Plane normals, unit vectors, interpolation variety |