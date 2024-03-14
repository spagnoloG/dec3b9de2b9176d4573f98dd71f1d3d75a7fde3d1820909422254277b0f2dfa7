import sys
import os

sys.path.append("/dust3r")

from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dl import ILoader


def ___setup_model(model_path: str, device: str):
    return load_model(model_path, device)


def ___set_args(
    model_path: str,
    device: str,
    batch_size: int,
    schedule: str,
    lr: float,
    niter: int,
    scene_graph: str,
):
    return {
        "model_path": model_path,
        "device": device,
        "batch_size": batch_size,
        "schedule": schedule,
        "lr": lr,
        "niter": niter,
        "scene_graph": scene_graph,
    }


def ___get_images(images: list, size: int) -> list:
    return load_images(images, size)


def ___run_inference(model, pairs: list, args: dict) -> dict:
    pairs = make_pairs(
        pairs, scene_graph=args["scene_graph"], prefilter=None, symmetrize=True
    )
    return inference(pairs, model, args["device"], batch_size=args["batch_size"])


def ___run_global_aligner(output: dict, args: dict) -> dict:
    scene = global_aligner(
        output, device=args["device"], mode=GlobalAlignerMode.PointCloudOptimizer
    )
    scene.compute_global_alignment(
        init="mst", niter=args["niter"], schedule=args["schedule"], lr=args["lr"]
    )
    return scene


def ___run_on_samples(samples: dict):
    trieste_samples = samples["Trieste"]
    model = ___setup_model(args["model_path"], args["device"])
    for i in range(0, 2):
        images = ___get_images(trieste_samples[i : i + 2], 512)
        output = ___run_inference(model, images, args)
        scene = ___run_global_aligner(output, args)
        return output, scene



if __name__ == "__main__":

    args = ___set_args(
        model_path="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        device="cuda",
        batch_size=1,
        schedule="cosine",
        lr=0.01,
        niter=300,
        scene_graph="complete",
    )
    dl = ILoader(dataset_dir="/datasets/drone_dataset")
    ___run_on_samples(dl.images_per_city)
