import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import pathlib
import shutil
import nrrd
import yaml
from skimage import exposure
from skimage.filters import threshold_otsu, threshold_local
from scipy.ndimage import binary_opening, label, gaussian_filter
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import MiniBatchNMF
import joblib
from scipy.optimize import nnls
import math


def display_all_nrrd_images(root_folder, max_cols=6):
    root_path = pathlib.Path(root_folder)
    nrrd_files = list(root_path.rglob("*.nrrd"))

    if not nrrd_files:
        print("No .nrrd files found.")
        return

    num_files = len(nrrd_files)
    num_cols = min(max_cols, num_files)
    num_rows = math.ceil(num_files / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 2 * num_rows))
    axes = axes.flatten()  # Makes indexing easier even if rows=1

    for i, file in enumerate(nrrd_files):
        try:
            img, _ = nrrd.read(str(file))
            rel_path = file.relative_to(root_path)
            tail_path = pathlib.Path(*rel_path.parts[-3:])

            ax = axes[i]
            ax.imshow(img, cmap='gray')
            ax.set_title(str(rel_path), fontsize=8)
            ax.axis('off')

        except Exception as e:
            print(f"Failed to display {file}: {e}")

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()