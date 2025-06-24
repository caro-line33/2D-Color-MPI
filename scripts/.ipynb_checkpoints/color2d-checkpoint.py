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

class Dataset():
    def __init__(self, pathstr):
        self.dir_path = pathlib.Path(pathstr)
        self.items = []

        if self.dir_path.exists():
            try:
                imgs_path = self.dir_path / "image_sets"
                for folder in imgs_path.iterdir():
                    self.items.append(DatasetItem(folder))
            except:
                print("Invalid folder structure")
                return
        else:    
            self.dir_path.mkdir(parents=True, exist_ok=True)
            (self.dir_path / "image_sets").mkdir(parents=True, exist_ok=True)
            self.items = []
            print(f"Created empty dataset at {self.dir_path}.")

    def append_data(self, *pathstrings):
        for pathstr in pathstrings:
            src = pathlib.Path(pathstr)
            if not src.exists():
                print(f"Path does not exist: {src}")
                continue
            dest = self.dir_path / "image_sets" / src.name
            if dest.exists():
                print(f"Folder {src.name} already exists in dataset. Skipping.")
                continue
            try:
                shutil.copytree(src, dest)
                new_item = DatasetItem(dest)
                self.items.append(new_item)
                print(f"Appended data from {src} → {dest}")
            except Exception as e:
                print(f"Failed to copy {src}: {e}")
            
    # def display_all(self):
        # for item in self.items:
            # print_all_info(item)

class ImageStack:
    def __init__(self, pathstr):
        self.pathstring = pathlib.Path(pathstr)
        self.amplitudes = self.get_amplitudes()
        self.image_matrix = self.get_normalized_images()

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
    
    def get_normalized_images(self):
        images = []
        for rf_amp_idx, amplitude in enumerate(self.amplitudes):
            load_file_path = (self.pathstring / "advanced_mpi_images" / "nrrd" 
                              / f"a{rf_amp_idx}_combined_projections" / "sharpened_collinear" 
                              / "0000.nrrd").resolve()
            img, _ = nrrd.read(str(load_file_path))
            images.append(img / amplitude)
        
        image_matrix = np.stack(images, axis=0) # shape: (num_amplitudes, H, W)

        return image_matrix # shape: (num_amplitudes, H, W)

    def foreground_mask(self, roi_coords: list =None):
        A, H, W = self.image_matrix.shape
        if not self.valid_roi(roi_coords):
            return
        binary_masks = []
        for i in range(self.image_matrix.shape[0]):
            layer = self.image_matrix[i]
            binary_mask = np.zeros((H, W), dtype=np.uint8)
            x1, x2, y1, y2 = roi_coords
            region = layer[y1:y2, x1:x2]
            smoothed = gaussian_filter(region, sigma=1.7)
            smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)
            img_eq = exposure.equalize_adapthist(smoothed, clip_limit=0.0045)
            global_thresh = 1.2 * threshold_otsu(img_eq)
            local_thresh = threshold_local(img_eq, block_size=33, offset=0.01)
            region_mask = ((smoothed > local_thresh) & (smoothed > global_thresh)).astype(np.uint8)
            cleaned = binary_opening(region_mask, structure=np.ones((3, 3)))
            binary_mask[y1:y2, x1:x2] = cleaned
            binary_masks.append(binary_mask)
        binary_stack = np.stack(binary_masks, axis=0)
        num_layers = binary_stack.shape[0]
        sum_mask = np.sum(binary_stack, axis=0)
        final_mask = (sum_mask > (num_layers / 2)).astype(np.uint8)
        return final_mask
    
    def get_matrix(self, target_area):
        if type(target_area) == list:
            if not self.valid_roi(target_area): return
            x1, x2, y1, y2 = target_area
            roi_matrix = self.image_matrix[:, y1:y2, x1:x2]
            roi_flat = roi_matrix.reshape(self.image_matrix.shape[0], -1)  
            return roi_flat
        if type(target_area) == np.ndarray:
            foreground_matrix = self.image_matrix[:, target_area != 0]
            return foreground_matrix
        return self.image_matrix.reshape(self.image_matrix.shape[0], -1)

    def display_images(self):
        captions = [f"{v}T" for v in self.amplitudes]
        A, H, W = self.image_matrix.shape
        fig, axes = plt.subplots(1, A, figsize=(4 * A, 4))
        for i, ax in enumerate(axes):
            ax.imshow(self.image_matrix[i], cmap='gray')
            ax.set_title(captions[i])
            ax.axis('on')
        plt.tight_layout()
        plt.show()
        print(f"{len(self.image_matrix)} images of dimension {H} x {W}")

    def visualize_vectors(self, roi_coords: list, mask: bool = True):
        if not self.valid_roi(roi_coords): return

        if mask:
            roi_matrix = self.get_matrix(self.foreground_mask(roi_coords))  # shape (A, N)
        else:
            roi_matrix = self.get_matrix(roi_coords)  # shape (A, N)

        # Normalize each vector (column) to unit norm
        norms = np.linalg.norm(roi_matrix, axis=0, keepdims=True) + 1e-8
        normed_matrix = roi_matrix / norms

        # Plot each vector
        plt.figure(figsize=(8, 5))
        for j in range(normed_matrix.shape[1]):
            plt.plot(self.amplitudes, normed_matrix[:, j], alpha=0.3)

        plt.xlabel("Amplitude")
        plt.ylabel("Normalized Signal")
        plt.title("Normalized ROI Vectors")
        plt.grid(True)
        plt.show()
    
    def valid_roi(self, coords):
        if len(coords) != 4: return False
        x1, x2, y1, y2 = coords
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if 0 <= x1 < x2 <= self.image_matrix.shape[2] and 0 <= y1 < y2 <= self.image_matrix.shape[1]: 
            coords = x1, x2, y1, y2
            return True
        else:
            print("invalid ROI")
            return False

class DatasetItem(ImageStack):
    def __init__(self, pathstr):
        super().__init__(pathstr)
        rois_path = self.pathstring / "rois.npz"
        if rois_path.exists():
            self.rois = dict(np.load(rois_path, allow_pickle=True))
        else:
            self.rois = self.identify_rois()
            np.savez(rois_path, self.rois, allow_pickle = True)
    
def colorize(image_path, dataset_path):
    images = ImageStack(image_path)
    image_matrix, image_amplitudes = images.image_matrix, images.amplitudes
    dataset = Dataset(dataset_path)
    dataset_amplitudes = dataset.amplitudes
    if not set(image_amplitudes).issubset(set(dataset_amplitudes)):
        print("Dataset does not contain sufficient amplitudes to colorize image")
        return

def str_to_coords(coords_str):
    coords = list(map(int, coords_str.split(",")))
    x1, x2, y1, y2 = coords
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [x1, x2, y1, y2]
        