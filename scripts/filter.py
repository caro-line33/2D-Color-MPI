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

def filter_images(root_folder):
    root_path = pathlib.Path(root_folder)
    nrrd_files = list(root_path.rglob("*.nrrd"))
    fft_img_data = []
    magnitudes = []
    for nrrd_file in nrrd_files:
        img, _ = nrrd.read(nrrd_file)
        F_o = np.fft.fft2(img)
        F = np.fft.fftshift(F_o)
        magnitude = np.abs(F)
        phase = np.angle(F)
        file_data = (img, magnitude, phase)
        fft_img_data.append(file_data)
        magnitudes.append(magnitude)
    magnitude_stack = np.stack(magnitudes, axis=0)
    filtered_magnitudes = np.min(magnitude_stack, axis=0)
    print_image_array(np.log1p(filtered_magnitudes), title="Log-scaled Minimum Magnitude")


def print_image_array(img_array, ax=None, min=None, max=None, title=' '):
    if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    if max == None and min == None:
        ax.imshow(img_array, cmap='gray')
    else:
        ax.imshow(img_array, cmap='gray', vmin=min, vmax=max)
    
