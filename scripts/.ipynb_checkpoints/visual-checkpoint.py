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

class ImageStack():
    def __init__(self, pathstr):
        self.pathstring = pathlib.Path(pathstr)
        self.amplitudes = self.get_amplitudes()
        self.image_matrix = self.get_normalized_images()

    # PROCESSING FUNCTIONS
    def get_amplitudes(self):
        with open(self.pathstring / "rf_amplitude_mapping.yaml", "r") as f:
            combined_rf_amp_dict = yaml.safe_load(f)

        x_amplitudes = list(combined_rf_amp_dict['tx_x'].values())
        z_amplitudes = list(combined_rf_amp_dict['tx_z'].values())

        if len(x_amplitudes) != len(z_amplitudes):
            raise ValueError(
                "Number of values differ between channels in amplitude mapping:\n"
                f"{combined_rf_amp_dict}")

        return x_amplitudes
    
# DISPLAY-RELATED FUNCTIONS
def display_all_nrrd_images(root_folder):
    root_path = pathlib.Path(root_folder)
    nrrd_files = list(root_path.rglob("*.nrrd"))  # Recursively find all .nrrd files

    for file in nrrd_files:
        try:
            img, _ = nrrd.read(str(file))
            rel_path = file.relative_to(root_path)  # For title

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img, cmap='gray')
            ax.set_title(str(rel_path))
            ax.axis('off')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Failed to display {file}: {e}")

def display_image(input_img, ax=None, title=None, y_label=None, x_label=None):
    if type(input_img) == np.ndarray:
        img = input_img
    if input_img.is_file():
        img, _ = nrrd.read(str(input_img))
    else:
        print("Cannot display invalid image format.")
    if ax is None: 
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(img)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.axis('on')
    ax.grid(True)

