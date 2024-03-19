import numpy as np


def __get_loss(scene, metadata_list: list) -> list:
    """
    Loss function to compute the projection error of images taken with the drone camera.

    set of imgs: T = {I1, I2, I3, ..., In}
    for I in T:
       check if transition results in the same 3D point

    scene:
        c1 c2 c3 c4 c5
        gt1 * scene1 ~ gt2
        gt2 * scene2 ~ gt3
        gt3 * scene3 ~ gt4
        gt4 * scene4 ~ gt5
        gt5 * scene5 ~ gt6

    WARN: metadata should then consist of [I1, I2, I3, ..., In + 1] where I1 is the first image in the sequence.
    """

    def setup_gt_rt_matrices(metadata_list: list):
        gt_rt_matrices = []
        for metadata in metadata_list:
            position = metadata["position"]
            rotation = metadata["rotation"]
            tx, ty, tz = position["x"], position["y"], position["z"]
            rot_x, rot_y, rot_z = rotation["x"], rotation["y"], rotation["z"]
            T = setup_translation_vector(tx, ty, tz)
            R = setup_rotation_mtx(rot_x, rot_y, rot_z)
            gt_rt_matrices.append(np.hstack((R, T)))
        return gt_rt_matrices

    scene_rt_matrices = scene.get_im_poses()
    gt_rt_matrices = setup_gt_rt_matrices(metadata_list)

    # Compute the projection error
    projection_erros = []
    for i in range(len(scene_rt_matrices)):
        scene_rt = scene_rt_matrices[i].cpu().detach().numpy()
        gt_rt = gt_rt_matrices[i]
        expected_rt = gt_rt_matrices[i + 1]
        projected_rt = gt_rt @ scene_rt
        projection_erros.append(np.linalg.norm(expected_rt - projected_rt))

    return projection_erros


def setup_translation_vector(tx: float, ty: float, tz: float):
    # T = | Tx |
    #     | Ty |
    #     | Tz |
    T = np.array([[tx], [ty], [tz]])
    return T


def setup_rotation_mtx(rot_x: float, rot_y: float, rot_z: float):
    # R_x = | 1  0       0      |
    #       | 0  cos(x) -sin(x) |
    #       | 0  sin(x)  cos(x) |
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(rot_x), -np.sin(rot_x)],
            [0, np.sin(rot_x), np.cos(rot_x)],
        ]
    )

    # R_y = |  cos(y)  0  sin(y) |
    #       |  0       1  0      |
    #       | -sin(y)  0  cos(y) |
    R_y = np.array(
        [
            [np.cos(rot_y), 0, np.sin(rot_y)],
            [0, 1, 0],
            [-np.sin(rot_y), 0, np.cos(rot_y)],
        ]
    )

    # R_z = | cos(z) -sin(z)  0 |
    #       | sin(z)  cos(z)  0 |
    #       | 0       0       1 |
    R_z = np.array(
        [
            [np.cos(rot_z), -np.sin(rot_z), 0],
            [np.sin(rot_z), np.cos(rot_z), 0],
            [0, 0, 1],
        ]
    )

    # Combine the rotation matrices
    R = R_x @ R_y @ R_z
    return R


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
