import torch
from PIL import Image
import os


class ILoader(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.cities = [
            "Ljubljana",
            "Venice",
            "MB",
            "Trieste",
            "Zagreb",
            "Graz",
            "Klagenfurt",
            "Udine",
            "Pula",
            "Pordenone",
            "Szombathely",
        ]
        self.images_per_city = {}

        self._set_all_image_paths()

    def _sort_array_by_digit(self, array: list) -> list:

        def __get_digit(string: str) -> int:
            return int(string.split("_")[-1].split(".")[0])

        return sorted(array, key=__get_digit)

    def _set_all_image_paths(self):
        self.all_image_paths = []
        for root, dirs, files in os.walk(self.dataset_dir):
            for file in files:
                if file.endswith(".jpeg"):
                    self.all_image_paths.append(os.path.join(root, file))

        print(f"Found {len(self.all_image_paths)} images in {self.dataset_dir}")

        images_per_city = {}

        for image in self.all_image_paths:
            for city in self.cities:
                if city in image:
                    if city in images_per_city:
                        images_per_city[city].append(image)
                    else:
                        images_per_city[city] = list()

        for city in images_per_city:
            images_per_city[city] = self._sort_array_by_digit(images_per_city[city])

        self.images_per_city = images_per_city

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> None:
        return None


if __name__ == "__main__":
    dataset = ILoader(dataset_dir="../datasets/drone_dataset/")
