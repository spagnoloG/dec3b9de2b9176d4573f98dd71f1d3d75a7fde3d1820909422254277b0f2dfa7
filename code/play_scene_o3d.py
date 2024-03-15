import open3d as o3d
import trimesh
import numpy as np


def vis_pointcloud_o3d(pointcloud, geom_visual):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)

    print("Vertex Colors:", geom_visual.vertex_colors)

    if hasattr(geom_visual, "vertex_colors") and len(geom_visual.vertex_colors) > 0:
        colors = np.array(geom_visual.vertex_colors)
        if colors.ndim == 2 and colors.shape[1] == 4:
            colors_normalized = colors[:, :3] / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors_normalized)

        print(colors.shape)

    o3d.visualization.draw_geometries([pcd])


def vis_mesh_o3d(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.faces)
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(triangles),
    )
    if mesh.visual.vertex_colors is not None:
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
            mesh.visual.vertex_colors[:, :3] / 255
        )  # Normalize colors
    o3d.visualization.draw_geometries([o3d_mesh])


if __name__ == "__main__":
    scene = trimesh.load("./results/reconstruction_results/scene.glb")

    if not scene.is_empty:
        for geom_name, geom in scene.geometry.items():
            print(f"Geometry: {geom_name}")
            if isinstance(geom, trimesh.points.PointCloud):
                print(f"Point Cloud with {len(geom.vertices)} points.")
                vis_pointcloud_o3d(geom.vertices, geom.visual)

            elif isinstance(geom, trimesh.Trimesh):
                print(
                    f"Mesh with {len(geom.vertices)} vertices and {len(geom.faces)} faces."
                )
                if (
                    hasattr(geom.visual, "vertex_colors")
                    and geom.visual.vertex_colors.size > 0
                ):
                    print("Mesh has vertex colors.")
                else:
                    print("Mesh does not have vertex colors.")
                vis_mesh_o3d(geom)
            print("----------")
    else:
        print("The scene is empty or does not contain readable geometries.")
