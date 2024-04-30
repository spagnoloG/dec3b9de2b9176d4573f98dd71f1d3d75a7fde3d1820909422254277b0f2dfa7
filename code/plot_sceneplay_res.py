import json
import matplotlib.pyplot as plt
import seaborn as sns


def plot_hisogram_of_means(means: list):
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.hist(means, bins=50)
    axs.set_title("Histogram of means")
    axs.set_xlabel("Mean")
    axs.set_ylabel("Frequency")
    plt.savefig("histogram_of_means.png")


if __name__ == "__main__":
    with open("./results_1710942687.json", "r") as f:
        results = json.load(f)

    mean_per_scene = []
    for k, v in results.items():
        if "scene" in k:
            scene_losses = v["scene_losses"]
            mean_per_scene.append(sum(scene_losses) / len(scene_losses))
    print("Mean per scene:", mean_per_scene)

    plot_hisogram_of_means(mean_per_scene)

    print("Mean of means:", sum(mean_per_scene) / len(mean_per_scene))

    print("Done!")
