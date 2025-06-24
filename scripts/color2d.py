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

class Dataset():
    def __init__(self, pathstr):
        self.dir_path = pathlib.Path(pathstr)
        self.amplitudes = None
        self.tracers = []

        if self.dir_path.exists():
            info_path = self.dir_path / "dataset_info.yaml"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    info_data = yaml.safe_load(f) or {}
                self.amplitudes = info_data.get('amplitudes')
                self.tracers = info_data.get('tracers', [])
            else:
                self._initialize_yaml()
        else:
            self.dir_path.mkdir(parents=True, exist_ok=True)
            self._initialize_yaml()

    def _initialize_yaml(self):
        self.amplitudes = None
        self.tracers = []
        info_dict = {'amplitudes': self.amplitudes, 'tracers': self.tracers}
        with open(self.dir_path / "dataset_info.yaml", 'w') as f:
            yaml.dump(info_dict, f)

    def append_data(self, input_pathstr):
        try:
            img_set = ImageStack(input_pathstr)
        except:
            print("Invalid image set path")
            return None

        info_path = self.dir_path / "dataset_info.yaml"
        with open(info_path, 'r') as f:
            info_dict = yaml.safe_load(f) or {}

        if info_dict.get('amplitudes') is None:    
            self.amplitudes = img_set.amplitudes
            info_dict['amplitudes'] = self.amplitudes
            with open(info_path, 'w') as f:
                yaml.dump(info_dict, f)
        else:
            if info_dict['amplitudes'] != img_set.amplitudes:
                print("Amplitude mismatch! Cannot append dataset.")
                return None

        tracers = img_set.identify_rois()
        print(f"Adding data for {len(tracers)} tracers: {', '.join(tracers.keys())}")

        for tracer_name, X_new in tracers.items():
            X_new = np.maximum(X_new, 0)
            tracer_name = tracer_name.lower()
            model_path = self.dir_path / f"{tracer_name}.pkl"

            if tracer_name in self.tracers:
                model = joblib.load(model_path)
                model.partial_fit(X_new)
                joblib.dump(model, model_path)
            else:
                add_new = input(f"{tracer_name} not in dataset. Add new? (Y/N) ").strip().lower()
                if add_new != 'y':
                    continue
                model = MiniBatchNMF(n_components=8, random_state=0, batch_size=100, max_iter=1000, l1_ratio=0.5)
                model.fit(X_new)
                joblib.dump(model, model_path)
                self.tracers.append(tracer_name)

        # update tracers list in YAML after adding any new tracers
        info_dict['tracers'] = self.tracers
        with open(info_path, 'w') as f:
            yaml.dump(info_dict, f)

    def colorize(self, input_path):
        if self.amplitudes == None:
            print("Cannot colorize using empty set.")
            return None
        else:
            imgs_set = ImageStack(input_path) # A, R, C
            imgs, img_amps = imgs_set.image_matrix, imgs_set.amplitudes
            self.print_info()
            if img_amps != self.amplitudes:
                print("Image and dataset amplitudes must match.")
                return None

            A, rows, columns = imgs.shape
            
            img_pixels = imgs.reshape(A, rows*columns)
            img_pixels = np.maximum(img_pixels, 0)
            pixels = img_pixels.T

            matrices = []
            tracer_matrix_lengths = []

            for tracer in self.tracers: 
                model = joblib.load(self.dir_path / f"{tracer}.pkl")  # safer with full path
                H = model.components_  # shape: (components, features)
                H_T = H.T  # transpose: (features, components)

                matrices.append(H_T)  # append to list
                tracer_matrix_lengths.append(H.shape[0]) 
            
            full_A_matrix = np.hstack(matrices) # features, total components
            print(full_A_matrix.shape) # debugging purposes

            starts = np.cumsum([0] + tracer_matrix_lengths[:-1])
            color_matrix = np.zeros((len(self.tracers), rows, columns), dtype=np.float32) # 1 map for each tracer

            for index, pixel in enumerate(pixels):
                x, rnorm = nnls(full_A_matrix, pixel)
                r = index // columns
                c = index % columns

                for i, length in enumerate(tracer_matrix_lengths):
                    start = starts[i]
                    end = start + length
                    tracer_sum = np.sum(x[start:end])
                    if tracer_sum < 0.05: tracer_sum = 0
                    color_matrix[i, r, c] = tracer_sum

            max_v = np.max(color_matrix)
            min_v = np.min(color_matrix)

            for map, tracer in zip(color_matrix, self.tracers):
                print_image_array(map, max=max_v, min=min_v, title=tracer)


    def print_info(self):
        print(f"Dataset contains matrices for {len(self.tracers)} tracers at {len(self.amplitudes)} amplitudes")
        print(f"Tracers: {', '.join(self.tracers)}")
        print(f"Amplitudes: {'mT, '.join(map(str, self.amplitudes))}")
            
class ImageStack:
    def __init__(self, pathstr):
        self.pathstring = pathlib.Path(pathstr)
        self.amplitudes = self.get_amplitudes()
        self.image_matrix = self.get_normalized_images() # shape: (num_amplitudes, H, W)

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
    
    def get_normalized_images(self): # shape: (num_amplitudes, H, W)
        images = []
        for rf_amp_idx, amplitude in enumerate(self.amplitudes):
            load_file_path = (self.pathstring / "advanced_mpi_images" / "nrrd" 
                              / f"a{rf_amp_idx}_combined_projections" / "sharpened_collinear" 
                              / "0000.nrrd").resolve()
            img, _ = nrrd.read(str(load_file_path))
            images.append(img / amplitude)
        
        image_matrix = np.stack(images, axis=0) # shape: (num_amplitudes, H, W)

        return image_matrix # shape: (num_amplitudes, H, W)

    def foreground_mask(self, roi_coords: list =None): # binary mask, H x W
        A, H, W = self.image_matrix.shape
        if not self.valid_roi(roi_coords):
            print("Invalid ROI coordinates")
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
    
    def get_matrix(self, target_area): # shape : (samples, amplitudes)
        if type(target_area) == list:
            if not self.valid_roi(target_area): return
            x1, x2, y1, y2 = target_area
            roi_matrix = self.image_matrix[:, y1:y2, x1:x2]
            roi_flat = roi_matrix.reshape(self.image_matrix.shape[0], -1)  
            return roi_flat.T
        
        if type(target_area) == np.ndarray:
            foreground_matrix = self.image_matrix[:, target_area != 0]
            return foreground_matrix.T
        return self.image_matrix.reshape(self.image_matrix.shape[0], -1).T # shape : (samples, amplitudes)
    
    def display_overlay(self, *target_areas, ax=None): # None
        mean_image = np.mean(self.image_matrix, axis=0)  # shape (H, W)
        H, W = mean_image.shape

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        ax.imshow(mean_image, cmap='gray')
        
        for target_area in target_areas:
            if isinstance(target_area, list):
                if not self.valid_roi(target_area):
                    return
                x1, x2, y1, y2 = target_area
                roi_mask = np.zeros((H, W), dtype=np.uint8)
                roi_mask[y1:y2, x1:x2] = 1
                overlay = roi_mask
            elif isinstance(target_area, np.ndarray):
                overlay = target_area
            else:
                print("Unsupported target_area format.")
                return

        ax.imshow(overlay, cmap='cool', alpha=0.4)
        ax.set_title("Selected ROI")
        ax.axis('on')

        if ax is None:
            plt.tight_layout()
            plt.show()

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

    def visualize_vectors(self, roi_coords: list, mask: bool = True, 
                          plot_color = 'green', show_background = False):
        
        if not self.valid_roi(roi_coords):
            return
        
        if mask: 
            roi_mask = self.foreground_mask(roi_coords)

        if show_background and mask:
            x1, x2, y1, y2 = roi_coords
            back_mask = np.zeros_like(roi_mask)
            back_mask[y1:y2, x1:x2] = 1 - roi_mask[y1:y2, x1:x2]
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            (ax_overlay, ax_plot), (back_overlay, back_plot) = axes
            self.display_overlay(back_mask, ax=back_overlay)
            back_overlay.set_title("Background")
            back_matrix = self.get_matrix(back_mask)
            self.plot_norm_matrix(back_matrix, ax=back_plot, plot_color = 'green')
            back_plot.set_title("Background Signal Vectors")

        else: fig, axes = plt.subplots(3, 2, figsize=(12, 9))
        (ax_overlay, ax_plot), (d1_plot, d1_abs_plot), (d2_plot, d2_abs_plot) = axes

        if mask:
            roi_matrix = self.get_matrix(roi_mask)  # shape (A, N)
            self.display_overlay(roi_mask, ax=ax_overlay)
        else:
            roi_matrix = self.get_matrix(roi_coords)  # shape (A, N)
            self.display_overlay(roi_coords, ax=ax_overlay)

        self.plot_norm_matrix(roi_matrix, ax=ax_plot, plot_color = 'green', show_boxplot=True)
        d1_matrix = np.gradient(roi_matrix, axis=0)
        self.plot_norm_matrix(d1_matrix, ax=d1_plot)
        self.plot_norm_matrix(np.abs(d1_matrix), ax=d1_abs_plot)
        d1_plot.set_xticks(self.amplitudes[1:-1])
        d1_abs_plot.set_xticks(self.amplitudes[1:-1])
        d1_plot.set_title("first derivative")
        d1_abs_plot.set_title("abs val first derivative")
        d2_matrix = np.gradient(d1_matrix, axis=0)
        self.plot_norm_matrix(d2_matrix, ax=d2_plot)
        self.plot_norm_matrix(np.abs(d2_matrix), ax=d2_abs_plot)
        d2_plot.set_xticks(self.amplitudes[2:-2])
        d2_abs_plot.set_xticks(self.amplitudes[2:-2])
        d2_plot.set_title("second derivative")
        d2_abs_plot.set_title("abs val second derivative")


        plt.tight_layout()
        plt.show()

    def plot_norm_matrix(self, datamatrix, ax=None, plot_color = 'green', show_boxplot=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        normed_matrix = datamatrix / (np.linalg.norm(datamatrix, axis=1, keepdims=True) + 1e-8)
        amps = np.linspace(self.amplitudes[0], self.amplitudes[-1], datamatrix.shape[1])
            
        for j in range(normed_matrix.shape[0]):
            ax.plot(normed_matrix[j], color=plot_color, alpha=0.3)
        
        if show_boxplot:
            ax.boxplot(normed_matrix, positions = (range(0, (len(self.amplitudes)))), notch=True)
            mean_vals = np.mean(normed_matrix, axis=0)  # mean at each amplitude
            ax.plot(mean_vals, color='Black', marker='x', linestyle='None', label='Mean')
            mean_vals = np.mean(normed_matrix, axis=0)
            
            std_vals = np.std(normed_matrix, axis=0)
            ax.plot(mean_vals, color='black', linestyle=None, marker='x', label='Mean')

        ax.set_title("Normalized Signal Vectors")
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Normalized Signal")
        ax.set_xticks(range(0, (len(self.amplitudes))))
        ax.set_xticklabels(self.amplitudes)
        ax.grid(True)

        if ax is None:
            plt.tight_layout()
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
        
    def identify_rois(self):
        A, H, W = self.image_matrix.shape
        self.display_images()
        coords_list = []
        input_rois = input("Enter ROI coordinates as x1,x2,y1,y2. Separate multiple ROIs with semicolons.")
        roi_strings = input_rois.split(";")

        for roi_str in roi_strings:
            if self.valid_roi(str_to_coords(roi_str)):
                coords_list.append(str_to_coords(roi_str))
            else:
                print(f"skipping invalid ROI: {roi_str}")

        if coords_list == []:
            foreground = self.foreground_mask()
        
        else: # foreground = self.foreground_mask(roi_coords = coords_list)
            foreground = np.zeros((H, W), dtype=np.uint8)
            for index, coords in enumerate(coords_list):
                foreground_chunk = self.foreground_mask(coords)
                foreground[foreground_chunk > 0] = 1

        self.display_overlay(foreground)
        plt.show()
        labeled_array, num_components = label(foreground)

        rois = {}
        for label_id in range(1, num_components + 1):
            mask = (labeled_array == label_id)
            # Display the specific ROI over the original projection
            if np.sum(mask) < 10:
                continue # Skip very small ROIs
            
            self.display_overlay(mask)
            plt.show()
            
            print(f"ROI #{label_id}")
            roi_name = input("Enter a name for this ROI (or press Enter to skip): ").strip()

            if roi_name == "":
                continue
            
            this_matrix = self.get_matrix(mask)
           
            if roi_name in rois:
                rois[roi_name] = np.concatenate((rois[roi_name], this_matrix), axis=0)
            else:
                rois[roi_name] = this_matrix

        return rois

def colorize(image_path):
    images = ImageStack(image_path)
    image_matrix, image_amplitudes = images.image_matrix, images.amplitudes
    A, rows, columns = image_matrix.shape
    img_pixels = image_matrix.reshape(A, rows*columns)
    pixels = img_pixels.T
    colorimage = np.zeros((rows, columns), dtype=np.uint8)
    norms = []
    for index, pixel in enumerate(pixels):
        r = index // columns
        c = index % columns
        norm_val = np.linalg.norm(pixel)
        norms.append(norm_val)

    # After the loop:
    norms = np.array(norms)
    norms_scaled = (norms / norms.max()) * 255
    norms_scaled_uint8 = norms_scaled.astype(np.uint8)

    # Reshape back to image shape
    colorimage = norms_scaled_uint8.reshape(rows, columns)

    # Now display:
    print_image_array(colorimage)

def print_image_array(img_array, ax=None, min=None, max=None, title=' '):
    if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    if max == None and min == None:
        ax.imshow(img_array, cmap='gray')
    else:
        ax.imshow(img_array, cmap='gray', vmin=min, vmax=max)

def str_to_coords(coords_str):
    coords = list(map(int, coords_str.split(",")))
    x1, x2, y1, y2 = coords
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [x1, x2, y1, y2]

