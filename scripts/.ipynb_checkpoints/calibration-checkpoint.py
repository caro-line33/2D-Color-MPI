import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import pathlib
from typing import Dict
import nrrd
import yaml
from numpy import ndarray
import skimage
from skimage import exposure
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import remove_small_objects, remove_small_holes
from scipy.ndimage import binary_opening, label, gaussian_filter
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class dataset:
    def __init__(self, dataset_path: str, 
                 calibration: bool = True,
                 colorize: bool = True, 
                 display_info: bool = False):
        self.display_info = display_info
        self.pathstring = pathlib.Path(dataset_path)
        self.amplitudes = get_amplitudes(dataset_path, display = self.display_info) 
            # x-axis amplitudes : list
        self.image_stack = get_normalized_images(dataset_path, display = self.display_info) 
            # shape: (num_amplitudes, H, W)
        if calibration:
            self.tracer_matrix_dictionary = identify_rois(dataset_path, save_rois = True)
            # {ROI name (str) : matrix with shape (amplitudes, # of vectors in ROI)}
            # all vectors are unit vectors

    
    def append_data(self, newdata_str):
        new_amplitudes = get_amplitudes(newdata_str)
        if new_amplitudes != self.amplitudes:
            raise ValueError(
                "Amplitudes do not match with original dataset:\n"
                f"{new_amplitudes}"
            )
        add_rois = identify_rois(newdata_str)
        for key, new_matrix in add_rois.items():
            if key in self.rois:
                # Concatenate along columns (amplitudes × pixels)
                self.rois[key] = np.concatenate([self.rois[key], new_matrix], axis=1)
            else:
                self.rois[key] = new_matrix
    
    def set_display_mode(self, on_off: bool):
        if on_off:
            self.display_info = True
        else:
            self.display_info = False

def get_amplitudes(pathstr: str, display=False):
    path = pathlib.Path(pathstr)
    with open(path / "rf_amplitude_mapping.yaml", "r") as f:
        combined_rf_amp_dict = yaml.safe_load(f)

    x_amplitudes = list(combined_rf_amp_dict['tx_x'].values())
    z_amplitudes = list(combined_rf_amp_dict['tx_z'].values())

    def print_amps():
        print("x axis amplitudes (T): " + ", ".join(str(v) for v in x_amplitudes))
        print("z axis amplitudes (T): " + ", ".join(str(v) for v in z_amplitudes))

    if len(x_amplitudes) != len(z_amplitudes):
        raise ValueError(
            "Number of values differ between channels in amplitude mapping:\n"
            f"{combined_rf_amp_dict}"
        )

    printed_warning = False
    if combined_rf_amp_dict["tx_x"] != combined_rf_amp_dict["tx_z"]:
        print("Warning: Values differ between channels in amplitude mapping.")
        print_amps()
        print("Processing will proceed using x-channel amplitudes.")
        printed_warning = True

    if display and not printed_warning:
        print_amps()

    return x_amplitudes

def get_normalized_images(pathstr, display = False):
        amplitudes = get_amplitudes(pathstr)
        images = []
        for rf_amp_idx, amplitude in enumerate(amplitudes):
            load_file_path = (
                    pathlib.Path(pathstr) / "advanced_mpi_images" / "nrrd"
                    / f"a{rf_amp_idx}_combined_projections" 
                    / "sharpened_collinear" / "0000.nrrd"
            ).resolve()
            img, _ = nrrd.read(str(load_file_path))
            images.append(img / amplitude) # normalize each image by amplitude
        
        image_matrix = np.stack(images, axis=0) # shape: (num_amplitudes, H, W)

        if display:
            captions = [f"{v}T" for v in amplitudes]
            fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
            for i, ax in enumerate(axes):
                ax.imshow(images[i], cmap='gray')
                ax.set_title(captions[i])
                ax.axis('on')
            plt.tight_layout()
            plt.show()
            print(f"{len(images)} images of dimension {images[0].shape}")
            print(f"Saved as matrix of dimension {image_matrix.shape}")

        return image_matrix # shape: (num_amplitudes, H, W)
  
def identify_rois(pathstr, 
                  save_rois: bool = True, 
                  manual_select: bool = False,
                  input_rois: bool = True):
    amplitudes = get_amplitudes(pathstr)
    stacked = get_normalized_images(pathstr, amplitudes)
    norm_stack = make_unit_vectors(stacked)
    mean_image = np.mean(stacked, axis=0)

    W = stacked.shape[2]
    H = stacked.shape[1]
    coords_list = [[0, W, 0, H]]


    if input_rois:
        plt.figure(figsize=(6, 6))
        plt.imshow(mean_image, cmap='gray')
        plt.title("Averaged image")
        plt.axis('on')
        plt.show()
        coords_list = []
        input_rois = input("Enter ROI coordinates as x1,x2,y1,y2. Separate multiple ROIs with semicolons.")
        roi_strings = input_rois.split(";")
        for roi_string in roi_strings:
            coords = list(map(int, roi_string.split(",")))
            x1, x2, y1, y2 = coords
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))

            if 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H:
                coords_list.append([x1, x2, y1, y2])
            else:
                print(f"Skipping out-of-bounds ROIs: {coords}")
         
    roi_mask = np.zeros((H, W), dtype=np.uint8)
    for x1, x2, y1, y2 in coords_list:
        roi_mask[y1:y2, x1:x2] = 1
    plt.figure(figsize=(6, 6))
    plt.imshow(mean_image, cmap='gray')
    plt.imshow(roi_mask, cmap='spring', alpha=0.4)
    plt.title("Analyzing selected regions")
    plt.axis('on')
    plt.show()
    

    final_mask = foreground_mask(pathstr, coords_list)
    labeled_array, num_components = label(final_mask)

    if not save_rois:
        return

    rois = {}
    for label_id in range(1, num_components + 1):
        mask = (labeled_array == label_id)
        # Display the specific ROI over the original projection
        if np.sum(mask) < 10:
            continue
        # Skip very small ROIs
        
        plt.figure(figsize=(6, 6))
        plt.imshow(mean_image, cmap='gray')
        plt.imshow(mask, cmap='spring', alpha=0.6)  # highlight in red/yellow
        plt.title(f"Label {label_id} - Highlighted ROI")
        plt.axis('off')
        plt.show()
        
        print(f"ROI #{label_id}")
        roi_name = input("Enter a name for this ROI (or press Enter to skip): ").strip()

        if roi_name == "":
            continue

        mask = (labeled_array == label_id)
        rows, cols = np.where(mask)
        vectors = [norm_stack[:, r, c] for r, c in zip(rows, cols)]  # shape: (A,) per vec

        roi_matrix = np.stack(vectors, axis=1)  # shape: (A, N)

        if roi_name in rois:
            rois[roi_name] = np.concatenate([rois[roi_name], roi_matrix], axis=1)
        else:
            rois[roi_name] = roi_matrix

    return rois

def foreground_mask(input_string, coords_list=None):
    image_stack = get_normalized_images(input_string)
    mean_image = np.mean(image_stack, axis=0)

    if coords_list is None:
        W = image_stack.shape[2]
        H = image_stack.shape[1]
        coords_list = [[0, W, 0, H]]
    else:
        H, W = image_stack.shape[1:3]  # needed to define shape for binary_mask

    binary_masks = []
    for i in range(image_stack.shape[0]):
        layer = image_stack[i]
        binary_mask = np.zeros((H, W), dtype=np.uint8)

        for coords in coords_list:
            x1, x2, y1, y2 = coords
            region = layer[y1:y2, x1:x2]
            smoothed = gaussian_filter(region, sigma=1.7)
            smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)
            img_eq = exposure.equalize_adapthist(smoothed, clip_limit=0.0045)
            global_thresh = 1.2 * threshold_otsu(img_eq)
            local_thresh = threshold_local(img_eq, block_size=33, offset=0.01)
            region_mask = ((smoothed > local_thresh) & (smoothed > global_thresh)).astype(np.uint8)
            cleaned = binary_opening(region_mask, structure=np.ones((3, 3)))

            # Insert cleaned ROI back into the full binary_mask
            binary_mask[y1:y2, x1:x2] = cleaned

        binary_masks.append(binary_mask)

    binary_stack = np.stack(binary_masks, axis=0)
    num_layers = binary_stack.shape[0]

    fig, axes = plt.subplots(1, num_layers, figsize=(4*num_layers, 4))

    for i in range(num_layers):
        layer = image_stack[i]
        mask = binary_masks[i]

        ax = axes[i] if num_layers > 1 else axes
        ax.imshow(layer, cmap='gray')
        ax.imshow(mask, cmap='cool', alpha=0.25)
        ax.set_title(f"Layer {i}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    sum_mask = np.sum(binary_stack, axis=0)
    final_mask = (sum_mask > (num_layers / 3)).astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(mean_image, cmap='gray')
    plt.imshow(final_mask, cmap='cool', alpha=0.2)
    plt.title("Foreground-background seperation over original image")
    plt.axis('on')
    plt.show()

    return final_mask


def make_unit_vectors(image_stack):
    A, H, W = image_stack.shape
    flat_pixels = image_stack.reshape(A, -1).T
    norms = np.linalg.norm(flat_pixels, axis=1, keepdims=True) + 1e-8
    unit_vectors = flat_pixels / norms
    normalized_stack = unit_vectors.T.reshape(A, H, W)
    return normalized_stack

def colorize(image_pathstr, data_pathstr, num_amps=None, num_tracers=None, match_threshold=0.98):
    image_amplitudes = get_amplitudes(image_pathstr)
    data_amplitudes = get_amplitudes(data_pathstr)

    if image_amplitudes != data_amplitudes:
        raise ValueError(
            "Amplitudes of datasets not compatible:\n"
            f"Images contain amplitudes: {image_amplitudes}\n"
            f"Data contains amplitudes: {data_amplitudes}"
        )

    image_stack = get_normalized_images(image_pathstr)  # shape: (A, H, W)
    tracer_data = identify_rois(data_pathstr) # {"roi name" : matrix (A, num_vectors_for_tracer)}
    print("Colorizing with:", ", ".join(tracer_data.keys()))
    
    signal_mask = foreground_mask(image_pathstr)
    
    a_matrix_list = []
    for roi_matrix in tracer_data.values():
        a_matrix_list.append(representative_vector(roi_matrix))
    a_matrix = np.stack(a_matrix_list)
    print(a_matrix.shape)
    print(a_matrix)

def representative_vector(roi_matrix):
    avg_vector = np.mean(roi_matrix, axis = 1)
    normed_vector = avg_vector / np.linalg.norm(avg_vector)
    return normed_vector





    