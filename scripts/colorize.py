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

class ImageStack:
# Usage: images_name = ImageStack(path_to_images)
# This class allows for processing and visualization of image sets

    def __init__(self, pathstr):
        self.pathstring = pathlib.Path(pathstr)
        self._amplitudes = self._get_amplitudes()
        self._image_matrix = self._get_normalized_images()

    def _get_amplitudes(self): # -> list of x-amplitudes from rf_amplitude_mapping.yaml
        with open(self.pathstring / "rf_amplitude_mapping.yaml", "r") as f:
            combined_rf_amp_dict = yaml.safe_load(f)

        x_amplitudes = list(combined_rf_amp_dict['tx_x'].values())
        z_amplitudes = list(combined_rf_amp_dict['tx_z'].values())

        if len(x_amplitudes) != len(z_amplitudes):
            raise ValueError(
                "Number of values differ between channels in amplitude mapping:\n"
                f"{combined_rf_amp_dict}")

        return x_amplitudes
    
    def _get_normalized_images(self): # -> 3d array, shape: (num_amplitudes, H, W)
        images = []
        for rf_amp_idx, amplitude in enumerate(self.amplitudes):
            load_file_path = (self.pathstring / "advanced_mpi_images" / "nrrd" 
                                / f"a{rf_amp_idx}_combined_projections" / "sharpened_collinear" 
                                / "0000.nrrd").resolve()
            
            if not load_file_path.exists():
                raise FileNotFoundError(f"NRRD file not found at {load_file_path}")
            
            img, _ = nrrd.read(str(load_file_path))
            images.append(img / amplitude)
        
        image_matrix = np.stack(images, axis=0)
        return image_matrix
    
    def display_images(self, display_all = False): # -> None
        captions = [f'{v}T' for v in self._amplitudes]
        A, H, W = self._image_matrix.shape
        fig, axes = plt.subplots(1, A, figsize=(4*A, 4))
        for i, ax in enumerate(axes):
            ax.imshow(self._image_matrix[i], cmap='gray')
            ax.set_title(captions[i])
            ax.axis('on')
        plt.tight_layout()
        plt.show()
        print(f"{len(self._image_matrix)} images of dimension {H} x {W}")
        return None