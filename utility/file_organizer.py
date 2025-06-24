import pathlib
import yaml
import shutil
import argparse

def organize(*input_paths: str) -> str:
    amplitude_path_dict = {get_amplitude(input_path): pathlib.Path(input_path) for input_path in input_paths}
    sorted_amplitude_dict = dict(sorted(amplitude_path_dict.items()))
    amplitudes_list = list(sorted_amplitude_dict.keys())
    paths = list(sorted_amplitude_dict.values())

    output_directory = get_directory("Enter the destination directory for the combined dataset")
    combined_name = input("Enter the name for the new combined dataset")

    dest_path = output_directory / combined_name
    dest_path.mkdir(parents=True, exist_ok=True)
    update_yaml_files(*input_paths, dest_root = dest_path, amplitudes=amplitudes_list)
    update_imaging_files(dest_path, paths)

    return str(dest_path)
    

def get_directory(prompt: str) -> pathlib.Path:
    path = pathlib.Path(input(f"{prompt}: "))
    while not path.is_dir():
        path = pathlib.Path(input(f"Invalid directory. Please enter a valid path: "))
    return path

def get_amplitude(input_path: str) -> int:
    pathstring = pathlib.Path(input_path)
    with open(pathstring / "rf_amplitude_mapping.yaml", "r") as f:
        combined_rf_amp_dict = yaml.safe_load(f)
        if combined_rf_amp_dict["tx_x"] != combined_rf_amp_dict["tx_z"]:
            raise ValueError(
                f"Values differ between channels in amplitude mapping. "
                f"{combined_rf_amp_dict}"
            )
        elif len(combined_rf_amp_dict["tx_x"]) != 1:
            raise ValueError(
                f"Invalid file format: image folders must contain 1 amplitude scans only. "
                f"{combined_rf_amp_dict}"
            )
        return list(combined_rf_amp_dict["tx_x"].values())[0]

def update_yaml_files(*input_paths, dest_root, amplitudes):

    new_key_params_path = dest_root / "key_parameters.yaml"
    old_key_params_path = pathlib.Path(input_paths[0]) / "key_parameters.yaml"

    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    with open(old_key_params_path, 'r') as f:
        old_key_params = yaml.safe_load(f)

    old_key_params["rf_amp"]["x"] = amplitudes
    old_key_params["rf_amp"]["z"] = amplitudes
    
    with open(new_key_params_path, 'w') as f:
        yaml.dump(old_key_params, f, Dumper=NoAliasDumper)


    rf_amplitude_path = dest_root / "rf_amplitude_mapping.yaml"
    
    rf_mapping = {
        'tx_x': {i: x for i, x in enumerate(amplitudes)},
        'tx_z': {i: z for i, z in enumerate(amplitudes)}
    }

    with open(rf_amplitude_path, 'w') as f:
        yaml.dump(rf_mapping, f)

def update_imaging_files(dest_root, paths):
    dest_nrrd_path = dest_root / "advanced_mpi_images" / "nrrd"
    dest_nrrd_path.mkdir(parents = True, exist_ok = True)
    dest_tiff_path = dest_root / "advanced_mpi_images" / "tiff"
    dest_tiff_path.mkdir(parents = True, exist_ok = True)

    for idx, path in enumerate(paths):
        for folder in ["combined_projections", "tx_x_projections", "tx_z_projections"]:
            shutil.copytree(path / "advanced_mpi_images" / "nrrd" / f"a0_{folder}",
                            dest_nrrd_path / f"a{idx}_{folder}",
                            dirs_exist_ok=True)
            shutil.copytree(path / "advanced_mpi_images" / "tiff" / f"a0_{folder}",
                            dest_tiff_path / f"a{idx}_{folder}",
                            dirs_exist_ok=True)