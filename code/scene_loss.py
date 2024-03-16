def __get_loss(scene, metadata_list: list) -> float:
    # TODO
    print(metadata_list)
    print(scene.get_intrinsics())
    return 0.0


#### CURRENT NOTES FOR IMPLEMENTATION

# -> We have lat, lon gt of the each image
# -> We want to reproject the centre of the image to the 3D point and then compare it with the gt 3D point
# -> This process is still unknown, must figure out how the geometries work

# # World cooridinate system: : A universal reference frame for all objects and cameras in the scene. It doesn't change with the movement of objects or cameras.
# # Camera coordinate system: The coordinate system of the camera. It is defined by the camera's intrinsic parameters and the camera's position and orientation in the world coordinate system.
#


# # Get the intrinsics of the camera
# # K =| f_x  0  c_x |
# #    |  0  f_y c_y |
# #    |  0   0   1  |
# # -> (f_x, f_y) -> focal length= distance between the camera sensor and lense center
# # -> (c_x, c_y) -> principal point= the point where the optical axis intersects the image plane
# print(scene.get_intrinsics())


# # Get camera c.s. to world c.s. transformation matrices
# # 4x4 matrices
# # | R11 R12 R13 Tx |
# # | R21 R22 R23 Ty |
# # | R31 R32 R33 Tz |
# # |  0   0   0   1 |
# # -> (R11, R12, R13), (R21, R22, R23), (R31, R32, R33) -> rotation matrix
# # -> (Tx, Ty, Tz) -> translation vector
# print(scene.get_im_poses())


# camera_cs_to_world_cs = scene.get_im_poses()
#
# # Get the actual principal points for each camera
# principal_points = scene.get_principal_points()
# intrinsic_matrices = scene.get_intrinsics()
#
# # Iterate over each transformation matrix and corresponding principal point
# for i, (proj_mtx, intr_mtx) in enumerate(
#     zip(camera_cs_to_world_cs, intrinsic_matrices)
# ):
#     proj_mtx_cpu = proj_mtx.cpu().detach().numpy()
#     intr_mtx_cpu = intr_mtx.cpu().detach().numpy()
#
#     x, y = principal_points[i].cpu().detach().numpy()
#     camera_point = np.linalg.inv(intr_mtx_cpu) @ np.array([x, y, 5])
#
#     world_point = proj_mtx_cpu @ np.append(camera_point, 1)
#
#     world_point = world_point / world_point[3]
#     print(f"Wold point for camera {i}: {world_point}")
#
