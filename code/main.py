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
from pyproj import Proj, transform


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


def __latlon_to_enu(lat, lon, alt, origin_lat, origin_lon, origin_alt):
    wgs84 = Proj(proj="latlong", datum="WGS84")
    ecef = Proj(proj="geocent", datum="WGS84")
    enu = Proj(proj="utm", zone=33, lat_0=origin_lat, lon_0=origin_lon, h_0=origin_alt)
    # Convert latlon to ECEF
    x, y, z = transform(wgs84, ecef, lon, lat, alt)
    # Convert ECEF to ENU
    x_enu, y_enu, z_enu = transform(ecef, enu, x, y, z)
    return x_enu, y_enu, z_enu


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

        image_list = [paths[i], paths[i + 1], paths[i + 2], paths[i + 3], paths[i + 4]]
        metadata_list = [
            metadata[i + 1],
            metadata[i + 2],
            metadata[i + 3],
            metadata[i + 4],
        ]
        for meta in metadata_list:
            lat, lon, alt = (
                meta["coordinate"]["latitude"],
                meta["coordinate"]["longitude"],
                150,
            )
            print(__latlon_to_enu(lat, lon, alt, origin_lat, origin_lon, origin_alt))

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
        loss = scene.compute_global_alignment(
            init="mst", niter=niter, schedule=schedule, lr=lr
        )

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
