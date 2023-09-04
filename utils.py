import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Plotting Utility functions
def plot_image(img):
    sns.heatmap(img, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.tight_layout()
    plt.show()


def plot_images(img1, img2, show_rmse=True, rescale_for_rmse=True, figsize=(12, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, img in zip(axes, [img1, img2]):
        sns.heatmap(img, cmap='viridis', ax=ax, xticklabels=False, yticklabels=False)

    fig.tight_layout()

    if show_rmse:
        if rescale_for_rmse:
            # Rescale the images
            [img1, img2] = [(x - x.min()) / (x.max() - x.min()) for x in [img1, img2]]
        rmse = np.sqrt(np.mean((img1 - img2) ** 2))
        plt.suptitle(f'RMSE: {rmse:.4f}', y=1.02)

    plt.show()
