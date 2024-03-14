import torch
from PIL import Image
import os

class ILoader(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self._set_all_image_paths()


    def _set_all_image_paths(self):
        self.all_image_paths = []
        with os.walk(self.dataset_dir) as (root, dirs, files):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    self.all_image_paths.append(os.path.join(root, file))

        print(f"Found {len(self.all_image_paths)} images in {self.dataset_dir}")

    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, index: int) -> None:
        return None


if __name__ == "__main__":
    dataset = ILoader("../../../datasets/drone_dataset/")
    print(dataset.all_image_paths)
