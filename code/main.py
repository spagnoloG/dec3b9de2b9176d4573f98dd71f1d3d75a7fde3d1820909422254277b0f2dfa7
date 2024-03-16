import sys
import os
import yaml
import numpy as np
from pprint import pprint
import os
import torch
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as pl
from pyproj import Proj, Transformer


sys.path.append("/dust3r")
from dust3r.inference import inference, load_model
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from dl import ILoader


def ___setup_model(model_path: str, device: str):
    return load_model(model_path, device)


def ___get_images(images: list, size: int) -> list:
    return load_images(images, size)


def ___run_inference(model, pairs: list, args: dict) -> dict:
    pairs = make_pairs(
        pairs,
        scene_graph=args["model"]["scene_graph"],
        prefilter=None,
        symmetrize=args["model"]["symmetrize"],
    )
    return inference(
        pairs, model, args["model"]["device"], batch_size=args["model"]["batch_size"]
    )


def ___run_global_aligner(output: dict, args: dict) -> dict:
    scene = global_aligner(
        output,
        device=args["global_alignment"]["device"],
        mode=GlobalAlignerMode.PointCloudOptimizer,
    )
    scene.compute_global_alignment(
        init=args["global_alignment"]["init"],
        niter=args["global_alignment"]["niter"],
        schedule=args["global_alignment"]["schedule"],
        lr=args["global_alignment"]["lr"],
    )
    return scene


def __latlon_to_utm(lat, lon):
    """
    Convert latitude and longitude to UTM coordinates.

    Returns:
    - (float, float, str): UTM easting, UTM northing, and UTM zone.
    """
    proj_utm = Proj(proj="utm", zone=int((lon + 180) / 6) + 1, datum="WGS84")
    transformer = Transformer.from_proj(Proj(proj="latlong", datum="WGS84"), proj_utm)
    easting, northing = transformer.transform(lon, lat)

    zone = int((lon + 180) / 6) + 1
    hemisphere = "N" if lat >= 0 else "S"
    utm_zone = f"{zone}{hemisphere}"

    return easting, northing, utm_zone


def ___run_on_samples(samples: dict, args: dict):
    trieste_samples = samples["Trieste"]
    model = ___setup_model(args["model"]["model_path"], args["model"]["device"])
    paths = [sample["path"] for sample in trieste_samples]
    metadata = [sample["metadata"] for sample in trieste_samples]

    for i in range(0, 2):
        # images = ___get_images(paths[i : i + 2], 512)
        # output = ___run_inference(model, images, args)
        # scene = ___run_global_aligner(output, args)
        # matches_im0, matches_im1 = ___get_matches(scene)
        origin = metadata[i]["coordinate"]
        origin_lat, origin_lon, origin_alt = (
            origin["latitude"],
            origin["longitude"],
            150,
        )
        print(origin)

        image_list = [paths[i], paths[i + 1], paths[i + 2]]
        metadata_list = [metadata[i], metadata[i + 1], metadata[i + 2]]
        # utm_coords = []
        # for meta in metadata_list:
        #    lat, lon, alt = (
        #        meta["coordinate"]["latitude"],
        #        meta["coordinate"]["longitude"],
        #        150,
        #    )
        #    utm_coords.append(__latlon_to_utm(lat, lon))

        print("image_list", image_list)
        print("metadata_list", metadata_list)

        winsize = max(1, (len(image_list) - 1) // 2)
        scene, outfile, imgs = __get_reconstructed_scene(
            args["reconstruct_scene"]["outdir"],
            model,
            args["reconstruct_scene"]["device"],
            args["reconstruct_scene"]["image_size"],
            image_list,
            args["reconstruct_scene"]["schedule"],
            args["reconstruct_scene"]["niter"],
            args["reconstruct_scene"]["min_conf_thr"],
            args["reconstruct_scene"]["as_pointcloud"],
            args["reconstruct_scene"]["mask_sky"],
            args["reconstruct_scene"]["clean_depth"],
            args["reconstruct_scene"]["transparent_cams"],
            args["reconstruct_scene"]["cam_size"],
            args["reconstruct_scene"]["scenegraph_type"],
            winsize,
            args["reconstruct_scene"]["refid"],
            args["reconstruct_scene"]["batch_size"],
        )
        return None


def __get_reconstructed_scene(
    outdir,
    model,
    device,
    image_size,
    filelist,
    schedule,
    niter,
    min_conf_thr,
    as_pointcloud,
    mask_sky,
    clean_depth,
    transparent_cams,
    cam_size,
    scenegraph_type,
    winsize,
    refid,
    batch_size,
):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    #
    # scene_graph:
    #                   -> complete: each image is connected to all the others and itself
    #                   -> swin: sliding window of size winsize
    #                   -> oneref: each image is connected to the first one
    #
    # symetrize:
    #                   -> if True, for each pair (i, j) we add (j, i)
    pairs = make_pairs(
        imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True
    )
    output = inference(pairs, model, device, batch_size=batch_size)

    mode = (
        GlobalAlignerMode.PointCloudOptimizer
        if len(imgs) > 2
        else GlobalAlignerMode.PairViewer
    )
    scene = global_aligner(output, device=device, mode=mode)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        print("running global aligner")
        loss = scene.compute_global_alignment(
            init="mst", niter=niter, schedule=schedule, lr=lr
        )

    # World cooridinate system: : A universal reference frame for all objects and cameras in the scene. It doesn't change with the movement of objects or cameras.
    # Camera coordinate system: The coordinate system of the camera. It is defined by the camera's intrinsic parameters and the camera's position and orientation in the world coordinate system.

    # Get the intrinsics of the camera
    # K =| f_x  0  c_x |
    #    |  0  f_y c_y |
    #    |  0   0   1  |
    # -> (f_x, f_y) -> focal length= distance between the camera sensor and lense center
    # -> (c_x, c_y) -> principal point= the point where the optical axis intersects the image plane
    print(scene.get_intrinsics())
    # Get camera c.s. to world c.s. transformation matrices
    # 4x4 matrices
    # | R11 R12 R13 Tx |
    # | R21 R22 R23 Ty |
    # | R31 R32 R33 Tz |
    # |  0   0   0   1 |
    # -> (R11, R12, R13), (R21, R22, R23), (R31, R32, R33) -> rotation matrix
    # -> (Tx, Ty, Tz) -> translation vector
    print(scene.get_im_poses())

    camera_cs_to_world_cs = scene.get_im_poses()

    # Get the actual principal points for each camera
    principal_points = scene.get_principal_points()
    intrinsic_matrices = scene.get_intrinsics()

    # Iterate over each transformation matrix and corresponding principal point
    for i, (proj_mtx, intr_mtx) in enumerate(
        zip(camera_cs_to_world_cs, intrinsic_matrices)
    ):
        proj_mtx_cpu = proj_mtx.cpu().detach().numpy()
        intr_mtx_cpu = intr_mtx.cpu().detach().numpy()

        x, y = principal_points[i].cpu().detach().numpy()
        camera_point = np.linalg.inv(intr_mtx_cpu) @ np.array([x, y, 1])

        world_point = proj_mtx_cpu @ np.append(camera_point, 1)

        world_point = world_point / world_point[3]
        print(f"Wold point for camera {i}: {world_point}")

    imgs = scene.imgs
    for img in imgs:
        print(img.shape)

    exit()

    outfile = __get_3D_model_from_scene(
        outdir,
        scene,
        min_conf_thr,
        as_pointcloud,
        mask_sky,
        clean_depth,
        transparent_cams,
        cam_size,
    )

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap("jet")
    depths_max = max([d.max() for d in depths])
    depths = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d / confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))

    return scene, outfile, imgs


def __get_3D_model_from_scene(
    outdir,
    scene,
    min_conf_thr=3,
    as_pointcloud=False,
    mask_sky=False,
    clean_depth=False,
    transparent_cams=False,
    cam_size=0.05,
):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return __convert_scene_output_to_glb(
        outdir,
        rgbimg,
        pts3d,
        msk,
        focals,
        cams2world,
        as_pointcloud=as_pointcloud,
        transparent_cams=transparent_cams,
        cam_size=cam_size,
    )


def __convert_scene_output_to_glb(
    outdir,
    imgs,
    pts3d,
    mask,
    focals,
    cams2world,
    cam_size=0.05,
    cam_color=None,
    as_pointcloud=False,
    transparent_cams=False,
):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(
            scene,
            pose_c2w,
            camera_edge_color,
            None if transparent_cams else imgs[i],
            focals[i],
            imsize=imgs[i].shape[1::-1],
            screen_width=cam_size,
        )

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("y", np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "scene.glb")
    print("(exporting 3D scene to", outfile, ")")
    scene.export(file_obj=outfile)
    return outfile


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    with open("conf.yml", "r") as file:
        args = yaml.safe_load(file)
    print("Args: ")
    args = args["main"]
    pprint(args)

    dl = ILoader(dataset_dir=args["dataset"]["dataset_path"])
    ___run_on_samples(dl.images_per_city, args)

#
# def ___get_matches(scene):
#    confidence_masks = scene.get_masks()
#    imgs = scene.imgs
#    focals = scene.get_focals()
#    poses = scene.get_im_poses()
#    pts3d = scene.get_pts3d()
#    pts2d_list, pts3d_list = [], []
#    for i in range(2):
#        conf_i = confidence_masks[i].cpu().numpy()
#        pts2d_list.append(
#            xy_grid(*imgs[i].shape[:2][::-1])[conf_i]
#        )  # imgs[i].shape[:2] = (H, W)
#        pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
#    reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
#    print(f"found {num_matches} matches")
#    matches_im1 = pts2d_list[1][reciprocal_in_P2]
#    matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]
#    return matches_im0, matches_im1
