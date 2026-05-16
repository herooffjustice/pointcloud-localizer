# PROMPTS.md

The most useful LLM prompts used during development of pointcloud-localizer.

## 1. Understanding the problem scope

"Explain the ICP algorithm step by step, including why we need KDTree for nearest neighbors, how SVD gives us the optimal rotation for point-to-point, and why point-to-plane can converge faster but requires normals."

This established the foundational understanding: ICP iterates between correspondence finding and transform estimation, KDTree makes correspondence search efficient, SVD gives a closed-form optimal rotation for fixed correspondences, and PtPl projects error onto surface normals for faster convergence but adds noise sensitivity.

## 2. Architecture and module design

"Design the file structure for pointcloud-localizer with modules: loader.py, synthetic.py, preprocess.py, icp.py, evaluate.py, cli.py. What should each module contain and what are the key function signatures?"

This prompt produced the initial architecture: data flow from loader → synthetic → preprocess → icp → evaluate, with cli.py orchestrating everything. The key insight was that synthetic.py must return ground-truth transforms alongside the point clouds, enabling evaluation.

## 3. SVD-based registration math

"Walk through the math for SVD-based point-to-point registration: given matched pairs (p_source, p_target), compute centroids, form cross-covariance H, do SVD, extract R and t. Also explain the reflection check (det < 0)."

This clarified the core algorithm: center both point sets to decouple rotation from translation, form H = q_s^T @ q_t, take SVD to get R = V @ U^T, and the det(R) < 0 check catches reflections where SVD returns an improper rotation.

## 4. Debugging PtPl divergence

"My point-to-plane ICP is oscillating — RMSE decreases in iteration 1 then increases in iteration 2. The linearized system seems correct. What could cause divergence?"

This identified that PtPl's linearized rotation can overshoot the correct transform, especially on symmetric surfaces. The solution was implementing a backtracking line search that tries progressively smaller steps (α = 0.5, 0.25, 0.125, 0.0625) and skips the iteration entirely if no step reduces RMSE.

## 5. Open3D compatibility and self-implementation

"open3d doesn't support Python 3.13. How can I implement point cloud I/O (PLY/PCD), mesh generation (torus, box+plane, sphere), surface sampling, and normal estimation using only numpy and scipy?"

This led to replacing all open3d dependencies: custom PLY/PCD parsers, parametric mesh generators, area-weighted surface sampling with barycentric coordinate interpolation, and PCA-based normal estimation using scipy KDTree. The challenge requirement "The ICP loop must be yours" made this the correct approach regardless.

## 6. Random transform generation

"scipy's Rotation.random() generates rotations up to 180 degrees regardless of a max_angle parameter. How do I generate a random rotation with a maximum angle constraint?"

This caught a critical bug: the initial implementation used Rotation.random() which could produce arbitrarily large rotations, making the robustness sweep's "small misalignment" cases sometimes receive 170° rotations. The fix was sampling a uniform angle in [0, max_rot_deg] and a random axis via normalized standard_normal, then using Rotation.from_rotvec(axis * angle).

## 7. Torus symmetry and ICP convergence

"ICP on a torus with 15+ degree initial misalignment gives significant rotation error even with zero noise. Is this a bug or expected behavior?"

This confirmed that rotational symmetry causes ICP to converge to local minima — the torus looks nearly identical under many rotations, so nearest-neighbor correspondences are ambiguous. The solution was using the asymmetric box_plane mesh (box sitting on a flat plane) for verification tests where ICP needs to converge reliably, while keeping the torus sweep to honestly demonstrate ICP's known limitations.

## 8. Line search for PtPl stability

"Point-to-plane ICP diverges on symmetric surfaces. How should I implement a backtracking line search that tries full step, then half, quarter, eighth steps, picking whichever reduces RMSE?"

This produced the current line search implementation: evaluate RMSE after the full PtPl step, then if it increases RMSE, try progressively smaller steps. If no step improves RMSE, set R_step = I and t_step = 0 (skip the iteration). A critical bug was found and fixed: the original code applied the bad full step even when line search found a smaller working step.