import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import pathlib
import nrrd
import yaml
from skimage import exposure
from skimage.filters import threshold_otsu, threshold_local
from scipy.ndimage import binary_opening, label, gaussian_filter
from sklearn.linear_model import LinearRegression

class Dataset:
    def __init__(self, dataset_path: str, display_info: bool = True):
        self.display_info = display_info
        self.pathstring = pathlib.Path(dataset_path)
        self.amplitudes = get_amplitudes(dataset_path, display = self.display_info) # list
        self.image_stack = get_normalized_images(dataset_path, display = self.display_info) # shape: (num_amplitudes, H, W)

        dict_path = self.pathstring / "full_dataset.npz"
        if dict_path.exists():
            self.full_dataset = dict(np.load(dict_path, allow_pickle=True))
        else:
            self.full_dataset = identify_rois(dataset_path, save_rois=True)
            np.savez(dict_path, **self.full_dataset)

        # init self.a_matrix, self.tracer_list, self.vector_dictionary
        self.update_cached_data()

    def append_data(self, newdata_str):
        new_amplitudes = get_amplitudes(newdata_str)
        if new_amplitudes != self.amplitudes:
            raise ValueError(
                "Amplitudes do not match with original dataset:\n"
                f"{new_amplitudes}"
            )
        add_rois = identify_rois(newdata_str)
        for key, new_matrix in add_rois.items():
            if key in self.full_dataset:
                # Concatenate along columns (amplitudes × pixels)
                self.full_dataset[key] = np.concatenate([self.full_dataset[key], new_matrix], axis=1)
            else:
                self.full_dataset[key] = new_matrix

        np.savez(self.pathstring / "full_dataset.npz", **self.full_dataset)
        self.update_cached_data()
    
    def update_cached_data(self):
        self.a_matrix = create_a_matrix(self.full_dataset)
        self.tracer_list = get_tracer_list(self.full_dataset)
        self.vector_dictionary = {
            name: representative_vector(mat)
            for name, mat in self.full_dataset.items()}
        
    def set_display_mode(self, on_off: bool):
        if on_off:
            self.display_info = True
        else:
            self.display_info = False

    def display_info(self):

        # Print all amplitudes
        print("amplitudes: " + ", ".join(str(v) for v in self.amplitudes))

        # Display all images from first dataset
        captions = [f"{v}T" for v in self.amplitudes]
        fig, axes = plt.subplots(1, len(self.image_stack), figsize=(4 * len(self.image_stack), 4))
        for i, ax in enumerate(axes):
            ax.imshow(self.image_stack[i], cmap='gray')
            ax.set_title(captions[i])
            ax.axis('on')
        plt.tight_layout()
        plt.show()
        print(f"{len(self.image_stack)} images of dimension {self.image_stack[0].shape}")
        print(f"Saved as matrix of dimension {self.image_stack.shape}")

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
    # norm_stack = make_unit_vectors(stacked)
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
        # changed norm_stack to stacked. COME BACK LATER!!
        vectors = [stacked[:, r, c] for r, c in zip(rows, cols)]  # shape: (A,) per vec

        roi_matrix = np.stack(vectors, axis=1)  # shape: (A, N)

        if roi_name in rois:
            rois[roi_name] = np.concatenate([rois[roi_name], roi_matrix], axis=1)
        else:
            rois[roi_name] = roi_matrix

    return rois

def foreground_mask(input_string, coords_list=None, display: bool = True):
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

def colorize(image_pathstr, data_pathstr, save_to = None):
    color_list = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
    image_amplitudes = get_amplitudes(image_pathstr)
    data_amplitudes = get_amplitudes(data_pathstr)

    if image_amplitudes != data_amplitudes:
        raise ValueError(
            "Amplitudes of datasets not compatible:\n"
            f"Images contain amplitudes: {image_amplitudes}\n"
            f"Data contains amplitudes: {data_amplitudes}"
        )

    image_stack = get_normalized_images(image_pathstr)  # shape: (A, H, W)
    tracer_dataset = Dataset(data_pathstr)
    tracers = tracer_dataset.tracer_list
    a_mat = tracer_dataset.a_matrix
    print("Colorizing with:", ", ".join(tracer_dataset.tracer_list))
    print(tracer_dataset.a_matrix)

    A, H, W = image_stack.shape

    X = image_stack.reshape(len(image_amplitudes), -1).T  # shape: (H*W, A)
    Y = a_mat.T                                           # shape: (K, A)

    model = LinearRegression(positive=True, fit_intercept=False)
    model.fit(Y.T, X.T)  # now both have shape (A, something)
    coef_flat = model.coef_.T  # shape: (H*W, K)
    coef_map = coef_flat.T.reshape(len(tracers), H, W)

def create_a_matrix(full_dataset):
    vectors = [representative_vector(mat) for mat in full_dataset.values()]  # each vec is (A,)
    a_matrix = np.stack(vectors, axis=1)  # shape: (A, K)
    return a_matrix

def get_tracer_list(full_dataset):
    tracer_list = list(full_dataset.keys())
    return tracer_list

def representative_vector(roi_matrix):
    avg_vector = np.mean(roi_matrix, axis = 1)
    normed_vector = avg_vector / np.linalg.norm(avg_vector)
    return normed_vector

def plot_vectors_2D(input_pathstr):
    data = Dataset(input_pathstr)
    amplitudes = np.array(data.amplitudes)
    cmap = cm.get_cmap("tab10")

    # ---- Overlay figure ----
    fig_overlay, ax_overlay = plt.subplots()

    # ---- Prepare separate row of subplots ----
    keys = list(data.full_dataset.keys())
    n_datasets = len(keys)
    fig_row, axes_row = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 4), squeeze=False)
    
    for i, key in enumerate(keys):
        value = np.array(data.full_dataset[key])  # shape: (A, N)
        A, N = value.shape

        magnitudes = np.linalg.norm(value, axis=0)
        min_mag = np.min(magnitudes)
        max_mag = np.max(magnitudes)
        range_mag = max_mag - min_mag if max_mag != min_mag else 1.0
        unit_vectors = value / (np.linalg.norm(value, axis=0, keepdims=True) + 1e-8)

        group_color = cmap(i % 10)

        # Overlay plot
        for j in range(N):
            alpha = (magnitudes[j] - min_mag) / range_mag
            ax_overlay.plot(amplitudes, unit_vectors[:, j], color=group_color, alpha=alpha)

        # Row subplot
        ax_row = axes_row[0, i]
        for j in range(N):
            alpha = (magnitudes[j] - min_mag) / range_mag
            ax_row.plot(amplitudes, unit_vectors[:, j], color=group_color, alpha=alpha)
        ax_row.set_title(f"{key}")
        ax_row.set_xlabel("Amplitude")
        ax_row.set_ylabel("Value")
        ax_row.grid(True)

    # Finalize overlay plot
    ax_overlay.set_xlabel("Amplitude")
    ax_overlay.set_ylabel("Normalized Vector Value")
    ax_overlay.set_title("Unit Vectors across Amplitudes (colored by dataset)")
    ax_overlay.grid(True)

    # Display all
    fig_row.tight_layout()
    fig_overlay.tight_layout()
    fig_row.show()
    fig_overlay.show()





    