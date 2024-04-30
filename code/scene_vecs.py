import numpy as np
import matplotlib.pyplot as plt
import torch


def __plot_vecs(scene_rt_matrices):
    # get rotation matrices

    fig, ax = plt.subplots()

    print(scene_rt_matrices)
    print(len(scene_rt_matrices))

    for mtx in scene_rt_matrices:
        # Extract translation vector (Tx, Ty, Tz)
        mtx = mtx.cpu().detach().numpy()
        t = mtx[:, 3]
        # Extract forward direction vector (Third column of rotation matrix)
        forward = mtx[:3, 2]

        # Plot the position as point
        ax.plot(t[0], t[1], "bo")

    # Set labels and title
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Camera Positions and Directions")
    ax.axis("equal")  # Ensure the aspect ratio is equal to see the correct direction

    plt.savefig("test.png")
    plt.close()


def __reproject_matrices(mtx_buff, scene_prev_rt_matrices, scene_curr_rt_matrices):
    origin_prev_mtx = scene_prev_rt_matrices[-1]
    inv_origin_prev_mtx = torch.linalg.inv(origin_prev_mtx)

    transformed_mtx_buff = []

    for mtx in scene_curr_rt_matrices:
        transformed_mtx = torch.matmul(inv_origin_prev_mtx, mtx)
        transformed_mtx_buff.append(transformed_mtx)

    mtx_buff.append(origin_prev_mtx)

    return scene_prev_rt_matrices
