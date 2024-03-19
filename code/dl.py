import torch
from PIL import Image
import os
import json


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
        self.image_metadata_dict = {}

        self._set_all_image_paths()

    def _get_metadata(self, path: str) -> None:
        """
        Extracts metadata from a JSON file and stores it in the metadata dictionary.
        """
        with open(path, newline="") as jsonfile:
            json_dict = json.load(jsonfile)
            path = path.split("/")[-1]
            path = path.replace(".json", "")
            self.image_metadata_dict[path] = json_dict["cameraFrames"]

    def _extract_info_from_filename(self, filename: str) -> tuple[str, int]:
        """
        Extracts information from the filename.
        """
        filename_without_ext = filename.replace(".jpeg", "")
        segments = filename_without_ext.split("/")
        info = segments[-1]
        try:
            number = int(info.split("_")[-1])
        except ValueError:
            print("Could not extract number from filename: ", filename)
            return "", 0

        info = "_".join(info.split("_")[:-1])

        return info, number

    def _sort_array_by_digit(self, array: list) -> list:
        def __get_digit(string: str) -> int:
            return int(string.split("_")[-1].split(".")[0])

        return sorted(array, key=__get_digit)

    def _set_all_image_paths(self):
        self.all_image_paths = []
        for root, dirs, files in os.walk(self.dataset_dir):
            if "Drone2Satellite" in root: # Vicos server specific
                continue
            for file in files:
                if file.endswith(".jpeg"):
                    self.all_image_paths.append(os.path.join(root, file))
                elif file.endswith(".json"):
                    self._get_metadata(os.path.join(root, file))

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

        for city in images_per_city:
            city_images = images_per_city[city]
            images_with_metadata = []
            for image in city_images:
                lookup_str, number = self._extract_info_from_filename(image)
                metadata = self.image_metadata_dict[lookup_str][number]
                images_with_metadata.append({"path": image, "metadata": metadata})
            images_per_city[city] = images_with_metadata

        self.images_per_city = images_per_city

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> None:
        return None


if __name__ == "__main__":
    dataset = ILoader(dataset_dir="../datasets/drone_dataset/")
