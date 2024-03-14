import sys
import os
import yaml
from pprint import pprint

sys.path.append("/dust3r")

from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
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


def ___run_on_samples(samples: dict, args: dict):
    trieste_samples = samples["Trieste"]
    model = ___setup_model(args["model"]["model_path"], args["model"]["device"])
    for i in range(0, 2):
        images = ___get_images(trieste_samples[i : i + 2], 512)
        output = ___run_inference(model, images, args)
        scene = ___run_global_aligner(output, args)
        matches_im0, matches_im1 = ___get_matches(scene)
        return output, scene


def ___get_matches(scene):
    confidence_masks = scene.get_masks()
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    pts2d_list, pts3d_list = [], []
    for i in range(2):
        conf_i = confidence_masks[i].cpu().numpy()
        pts2d_list.append(
            xy_grid(*imgs[i].shape[:2][::-1])[conf_i]
        )  # imgs[i].shape[:2] = (H, W)
        pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
    reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
    print(f"found {num_matches} matches")
    matches_im1 = pts2d_list[1][reciprocal_in_P2]
    matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]
    return matches_im0, matches_im1


if __name__ == "__main__":
    # Read yaml into dictionary
    with open("conf.yml", "r") as file:
        args = yaml.safe_load(file)
    print("Args: ")
    args = args["main"]
    pprint(args)

    dl = ILoader(dataset_dir=args["dataset"]["dataset_path"])
    ___run_on_samples(dl.images_per_city, args)
