import numpy as np
from scipy.spatial.transform import Rotation

from pointcloud_localizer.loader import make_transform, TriangleMesh, save_pointcloud


def _create_torus(R=1.0, r=0.3, n_radial=32, n_tube=16):
    vertices = []
    triangles = []
    for i in range(n_radial):
        theta = 2 * np.pi * i / n_radial
        for j in range(n_tube):
            phi = 2 * np.pi * j / n_tube
            x = (R + r * np.cos(phi)) * np.cos(theta)
            y = (R + r * np.cos(phi)) * np.sin(theta)
            z = r * np.sin(phi)
            vertices.append([x, y, z])

    for i in range(n_radial):
        for j in range(n_tube):
            v0 = i * n_tube + j
            v1 = i * n_tube + (j + 1) % n_tube
            v2 = ((i + 1) % n_radial) * n_tube + j
            v3 = ((i + 1) % n_radial) * n_tube + (j + 1) % n_tube
            triangles.append([v0, v2, v1])
            triangles.append([v1, v2, v3])

    return TriangleMesh(np.array(vertices), np.array(triangles))


def _create_box_plane():
    vertices = [
        [-1.0, -0.25, -1.0], [1.0, -0.25, -1.0], [1.0, 0.25, -1.0], [-1.0, 0.25, -1.0],
        [-1.0, -0.25, 1.0], [1.0, -0.25, 1.0], [1.0, 0.25, 1.0], [-1.0, 0.25, 1.0],
        [-2.0, -0.26, -2.0], [2.0, -0.26, -2.0], [2.0, -0.26, 2.0], [-2.0, -0.26, 2.0],
    ]
    triangles = [
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2],
        [8, 10, 9], [8, 11, 10],
    ]
    return TriangleMesh(np.array(vertices, dtype=np.float64), np.array(triangles, dtype=np.int64))


def _create_sphere(radius=0.5, n_lat=20, n_lon=20):
    vertices = []
    triangles = []
    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat
        for j in range(n_lon + 1):
            phi = 2 * np.pi * j / n_lon
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            vertices.append([x, y, z])

    for i in range(n_lat):
        for j in range(n_lon):
            v0 = i * (n_lon + 1) + j
            v1 = v0 + 1
            v2 = (i + 1) * (n_lon + 1) + j
            v3 = v2 + 1
            triangles.append([v0, v2, v1])
            triangles.append([v1, v2, v3])

    return TriangleMesh(np.array(vertices), np.array(triangles, dtype=np.int64))


def generate_default_mesh(mesh_type="torus"):
    if mesh_type == "torus":
        mesh = _create_torus()
    elif mesh_type == "box_plane":
        mesh = _create_box_plane()
    elif mesh_type == "sphere":
        mesh = _create_sphere()
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}. Use 'torus', 'box_plane', or 'sphere'.")
    mesh.compute_vertex_normals()
    return mesh


def sample_mesh(mesh, n_points=10000, seed=None):
    rng = np.random.default_rng(seed)
    triangles = mesh.triangles
    vertices = mesh.vertices

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0

    cross_products = np.cross(edge1, edge2)
    areas = np.linalg.norm(cross_products, axis=1) / 2.0
    areas = np.maximum(areas, 0)

    probabilities = areas / areas.sum()
    triangle_indices = rng.choice(len(triangles), size=n_points, p=probabilities)

    r1 = rng.random(n_points)
    r2 = rng.random(n_points)

    sqrt_r1 = np.sqrt(r1)
    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = sqrt_r1 * r2

    selected_v0 = v0[triangle_indices]
    selected_edge1 = edge1[triangle_indices]
    selected_edge2 = edge2[triangle_indices]

    points = selected_v0 + u[:, np.newaxis] * selected_edge1 + v[:, np.newaxis] * selected_edge2

    normals_data = None
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
        n0 = mesh.vertex_normals[triangles[:, 0]]
        n1 = mesh.vertex_normals[triangles[:, 1]]
        n2 = mesh.vertex_normals[triangles[:, 2]]
        selected_n0 = n0[triangle_indices]
        selected_n1 = n1[triangle_indices]
        selected_n2 = n2[triangle_indices]
        normals_data = w[:, np.newaxis] * selected_n0 + u[:, np.newaxis] * selected_n1 + v[:, np.newaxis] * selected_n2
        norm = np.linalg.norm(normals_data, axis=1, keepdims=True)
        norm[norm < 1e-12] = 1.0
        normals_data = normals_data / norm

    return points, normals_data


def add_noise(points, sigma=0.0, seed=None):
    if sigma <= 0:
        return points.copy(), None
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=points.shape)
    return points + noise, noise


def random_transform(max_rot_deg=30.0, max_trans=0.1, seed=None):
    rng = np.random.default_rng(seed)
    angle = np.deg2rad(rng.uniform(0, max_rot_deg))
    axis_raw = rng.standard_normal(3)
    axis = axis_raw / np.linalg.norm(axis_raw)
    R = Rotation.from_rotvec(axis * angle).as_matrix()
    t = rng.uniform(-max_trans, max_trans, size=3)
    return R, t


def apply_transform(points, R, t):
    return (points @ R.T) + t


def generate_scene(
    mesh=None,
    mesh_type="torus",
    n_points=10000,
    noise_sigma=0.0,
    max_rot_deg=30.0,
    max_trans=0.1,
    seed=None,
    source_path=None,
    target_path=None,
):
    if mesh is None:
        mesh = generate_default_mesh(mesh_type)

    seed_sample = seed if seed is None else seed * 3 + 0
    seed_transform = seed if seed is None else seed * 3 + 1
    seed_noise = seed if seed is None else seed * 3 + 2

    source, source_normals = sample_mesh(mesh, n_points, seed=seed_sample)

    R, t = random_transform(max_rot_deg, max_trans, seed=seed_transform)

    target_clean = apply_transform(source, R, t)
    target, noise = add_noise(target_clean, sigma=noise_sigma, seed=seed_noise)

    target_normals = None
    if source_normals is not None:
        target_normals = (source_normals @ R.T)
        norm = np.linalg.norm(target_normals, axis=1, keepdims=True)
        norm[norm < 1e-12] = 1.0
        target_normals = target_normals / norm

    result = {
        "source": source,
        "target": target,
        "source_normals": source_normals,
        "target_normals": target_normals,
        "R_gt": R,
        "t_gt": t,
        "T_gt": make_transform(R, t),
        "noise_sigma": noise_sigma,
        "max_rot_deg": max_rot_deg,
        "max_trans": max_trans,
    }

    if source_path is not None:
        save_pointcloud(source, source_path)
    if target_path is not None:
        save_pointcloud(target, target_path)

    return result