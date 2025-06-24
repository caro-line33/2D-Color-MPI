import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import pathlib
import shutil
import nrrd
import yaml
from scipy.fft import fft2, fftshift
from skimage import exposure
from skimage.filters import threshold_otsu, threshold_local
from scipy.ndimage import binary_opening, label, gaussian_filter
from sklearn.linear_model import LinearRegression
import h5py
import pathlib
import math

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

def explore_hdf(file_path):
    with h5py.File(file_path, 'r') as f:
        print(f"Exploring HDF5 file: {file_path}\n")
        
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"{name}: Dataset, shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{name}: Group")

        f.visititems(visitor)

def analyze_fft(file_path):
    # Load 2D image directly from NRRD file
    image, _ = nrrd.read(file_path)
    
    if image.ndim != 2:
        print("Error: This function expects a 2D NRRD file.")
        return
    
    # Compute 2D FFT
    F = fftshift(fft2(image))
    magnitude = np.abs(F)
    phase = np.angle(F)
    
    # Plot original, magnitude, and phase
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(np.log1p(magnitude), cmap='gray')
    axes[1].set_title("Magnitude (log scale)")
    axes[1].axis('off')
    
    axes[2].imshow(phase, cmap='twilight')
    axes[2].set_title("Phase")
    axes[2].axis('off')
    
    plt.show()



        


