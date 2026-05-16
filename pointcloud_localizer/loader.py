import numpy as np
import struct


def load_ply(path):
    points = []
    normals = []
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if b"end_header" in line:
                break

        has_normals = b"property float nx" in header or b"property float32 nx" in header

        lines = header.decode("ascii").split("\n")
        n_verts = 0
        is_binary = b"binary_little_endian" in header
        for line in lines:
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])

        if is_binary:
            if has_normals:
                dtype = np.dtype([(f"f{i}", np.float32) for i in range(6)])
            else:
                dtype = np.dtype([(f"f{i}", np.float32) for i in range(3)])

            data = np.frombuffer(f.read(n_verts * dtype.itemsize), dtype=dtype)
            points = np.column_stack([data[f"f{i}"].astype(np.float64) for i in range(3)])
            if has_normals:
                normals = np.column_stack([data[f"f{i}"].astype(np.float64) for i in range(3, 6)])
        else:
            points = []
            normals = []
            for _ in range(n_verts):
                line = f.readline().decode("ascii").strip()
                vals = [float(x) for x in line.split()]
                points.append(vals[:3])
                if has_normals and len(vals) >= 6:
                    normals.append(vals[3:6])
            points = np.array(points, dtype=np.float64)
            if has_normals and normals:
                normals = np.array(normals, dtype=np.float64)

    if len(normals) == 0:
        return points, None
    return points, normals


def load_pcd(path):
    points = []
    with open(path, "r") as f:
        n_points = 0
        data_start = False
        for line in f:
            if line.startswith("POINTS"):
                n_points = int(line.split()[-1])
            if line.strip() == "DATA ascii":
                data_start = True
                continue
            if data_start:
                vals = line.strip().split()
                if len(vals) >= 3:
                    points.append([float(vals[0]), float(vals[1]), float(vals[2])])
    return np.array(points, dtype=np.float64)


def load_pointcloud(path):
    ext = path.lower().split(".")[-1]
    if ext == "ply":
        pts, _ = load_ply(path)
        return pts
    elif ext == "pcd":
        return load_pcd(path)
    else:
        raise ValueError(f"Unsupported format: {ext}. Use .ply or .pcd")


def save_ply(points, path, normals=None):
    n = points.shape[0]
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if normals is not None:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        f.write("end_header\n")
        for i in range(n):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            if normals is not None:
                line += f" {normals[i, 0]:.6f} {normals[i, 1]:.6f} {normals[i, 2]:.6f}"
            f.write(line + "\n")


def save_pcd(points, path, normals=None):
    n = points.shape[0]
    with open(path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z")
        if normals is not None:
            f.write(" normal_x normal_y normal_z")
        f.write("\n")
        f.write("SIZE 4 4 4")
        if normals is not None:
            f.write(" 4 4 4")
        f.write("\n")
        f.write("TYPE F F F")
        if normals is not None:
            f.write(" F F F")
        f.write("\n")
        f.write("COUNT 1 1 1")
        if normals is not None:
            f.write(" 1 1 1")
        f.write("\n")
        f.write(f"WIDTH {n}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n}\n")
        f.write("DATA ascii\n")
        for i in range(n):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            if normals is not None:
                line += f" {normals[i, 0]:.6f} {normals[i, 1]:.6f} {normals[i, 2]:.6f}"
            f.write(line + "\n")


def save_pointcloud(points, path, normals=None):
    ext = path.lower().split(".")[-1]
    if ext == "ply":
        save_ply(points, path, normals)
    elif ext == "pcd":
        save_pcd(points, path, normals)
    else:
        save_ply(points, path + ".ply", normals)


class TriangleMesh:
    def __init__(self, vertices, triangles):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int64)

    def compute_vertex_normals(self):
        v0 = self.vertices[self.triangles[:, 0]]
        v1 = self.vertices[self.triangles[:, 1]]
        v2 = self.vertices[self.triangles[:, 2]]

        e1 = v1 - v0
        e2 = v2 - v0
        face_normals = np.cross(e1, e2)
        face_normals /= (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-12)

        vertex_normals = np.zeros_like(self.vertices)
        np.add.at(vertex_normals, self.triangles[:, 0], face_normals)
        np.add.at(vertex_normals, self.triangles[:, 1], face_normals)
        np.add.at(vertex_normals, self.triangles[:, 2], face_normals)

        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        vertex_normals /= norms
        self.vertex_normals = vertex_normals
        return vertex_normals


def load_mesh(path):
    ext = path.lower().split(".")[-1]
    if ext == "ply":
        return _load_mesh_ply(path)
    elif ext == "obj":
        return _load_mesh_obj(path)
    elif ext == "stl":
        return _load_mesh_stl(path)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")


def _load_mesh_obj(path):
    vertices = []
    faces = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                face_verts = []
                for p in parts[1:]:
                    idx = int(p.split("/")[0]) - 1
                    face_verts.append(idx)
                if len(face_verts) >= 3:
                    faces.append(face_verts[:3])

    return TriangleMesh(np.array(vertices), np.array(faces))


def _load_mesh_ply(path):
    vertices = []
    faces = []
    with open(path, "r") as f:
        header_ended = False
        is_binary = False
        n_verts = 0
        n_faces = 0

        while not header_ended:
            line = f.readline().strip()
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])
            elif line.startswith("format"):
                if "binary" in line:
                    is_binary = True
            elif line == "end_header":
                header_ended = True

        if is_binary:
            remaining = f.read()
            offset = 0
            for _ in range(n_verts):
                coords = struct.unpack_from("fff", remaining, offset)
                vertices.append(list(coords))
                offset += 12
            for _ in range(n_faces):
                n_tri = struct.unpack_from("B", remaining, offset)[0]
                offset += 1
                face = []
                for _ in range(n_tri):
                    idx = struct.unpack_from("i", remaining, offset)[0]
                    face.append(idx)
                    offset += 4
                if len(face) >= 3:
                    faces.append(face[:3])
        else:
            for _ in range(n_verts):
                line = f.readline().strip().split()
                vertices.append([float(line[0]), float(line[1]), float(line[2])])
            for _ in range(n_faces):
                line = f.readline().strip().split()
                n_verts_in_face = int(line[0])
                face = [int(line[i + 1]) for i in range(n_verts_in_face)]
                if len(face) >= 3:
                    faces.append(face[:3])

    mesh = TriangleMesh(np.array(vertices), np.array(faces))
    mesh.compute_vertex_normals()
    return mesh


def _load_mesh_stl(path):
    triangles = []
    vertices_list = []
    vertex_map = {}

    with open(path, "r") as f:
        normals = []
        verts = []
        reading = False
        for line in f:
            line = line.strip()
            if line.startswith("facet normal"):
                parts = line.split()
                normals.append([float(parts[2]), float(parts[3]), float(parts[4])])
                reading = True
                verts = []
            elif line.startswith("vertex") and reading:
                parts = line.split()
                v = (float(parts[1]), float(parts[2]), float(parts[3]))
                verts.append(v)
            elif line.startswith("endfacet"):
                face = []
                for v in verts:
                    if v not in vertex_map:
                        vertex_map[v] = len(vertices_list)
                        vertices_list.append(list(v))
                    face.append(vertex_map[v])
                triangles.append(face)

    return TriangleMesh(np.array(vertices_list), np.array(triangles))


def make_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def decompose_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t