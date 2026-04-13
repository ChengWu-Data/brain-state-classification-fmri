import os
import json
from pathlib import Path
from itertools import product

import numpy as np
import nibabel as nib
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score


# =========================
# User-configurable paths
# =========================
# Put this script in the project folder.
# After extracting sub-01.rar, make sure the extracted folder is inside DATA_DIR
# or point DATA_DIR to the correct location.

DATA_DIR = Path("./sub-01")       # folder containing extracted ses-test / ses-retest files
LABEL_PATH = Path("./label.mat")  # path to label.mat
OUTPUT_DIR = Path("./outputs")    # results / masks will be saved here

# Parameter search space
THRESHOLDS = [0.12, 0.15, 0.18, 0.20, 0.22, 0.24, 0.26]
K_VALUES = [6, 8, 10]
PCA_COMPONENTS = [None]
C_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
KERNELS = ["linear"]
GAMMA_VALUES = ["scale"]

RANDOM_STATE = 42


# =========================
# Utility functions
# =========================
def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def load_labels(label_path: Path) -> np.ndarray:
    """
    Load labels from label.mat.
    Expects a variable named 'label' with 184 labels.

    label mapping from uploaded readme:
    1 = Rest
    2 = Finger movement
    3 = Lips movement
    4 = Foot movement
    """
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    mat = loadmat(label_path)
    if "label" not in mat:
        raise KeyError("Variable 'label' was not found in label.mat")

    labels = np.asarray(mat["label"]).ravel().astype(int)

    if labels.ndim != 1:
        raise ValueError(f"Labels should be 1D after flattening, got shape {labels.shape}")

    if len(labels) != 184:
        raise ValueError(f"Expected 184 labels, got {len(labels)}")

    return labels


def find_nifti_files(data_dir: Path) -> dict:
    """
    Recursively search for .nii or .nii.gz files and identify
    ses-test and ses-retest datasets from file/folder names.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"DATA_DIR not found: {data_dir}\n"
            f"Please extract sub-01.rar first and update DATA_DIR if needed."
        )

    nii_files = list(data_dir.rglob("*.nii")) + list(data_dir.rglob("*.nii.gz"))

    if not nii_files:
        raise FileNotFoundError(
            f"No .nii or .nii.gz files found under {data_dir}. "
            f"Please extract sub-01.rar first."
        )

    dataset_paths = {"ses-test": None, "ses-retest": None}

    for f in nii_files:
        path_lower = str(f).lower()
        if "ses-test" in path_lower and dataset_paths["ses-test"] is None:
            dataset_paths["ses-test"] = f
        elif "ses-retest" in path_lower and dataset_paths["ses-retest"] is None:
            dataset_paths["ses-retest"] = f

    missing = [k for k, v in dataset_paths.items() if v is None]
    if missing:
        raise FileNotFoundError(
            f"Could not find files for: {missing}\n"
            f"Found NIfTI files:\n" + "\n".join(str(f) for f in nii_files)
        )

    return dataset_paths


def load_nifti_4d(nifti_path: Path):
    """
    Load 4D fMRI data from a NIfTI file.
    Returns:
        img: nibabel image object
        data: numpy array with shape (X, Y, Z, T)
    """
    img = nib.load(str(nifti_path))
    data = img.get_fdata()

    if data.ndim != 4:
        raise ValueError(f"Expected 4D fMRI data, got shape {data.shape} from {nifti_path}")

    return img, data


def create_brain_mask(data_4d: np.ndarray, threshold_ratio: float) -> np.ndarray:
    """
    Create a simple brain mask using a single threshold on the mean image.
    threshold = threshold_ratio * max(mean_image)
    Returns a boolean 3D mask.
    """
    mean_image = np.mean(data_4d, axis=3)
    threshold_value = threshold_ratio * np.max(mean_image)
    mask = mean_image > threshold_value

    if np.sum(mask) == 0:
        raise ValueError(
            f"Mask is empty for threshold_ratio={threshold_ratio}. "
            f"Try a lower threshold."
        )

    return mask


def save_mask_nifti(mask: np.ndarray, reference_img, save_path: Path) -> None:
    """
    Save 3D brain mask as a NIfTI file using the reference image affine/header.
    """
    mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine=reference_img.affine, header=reference_img.header)
    nib.save(mask_img, str(save_path))


def extract_masked_features(data_4d: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a 3D brain mask to 4D fMRI data and reshape to (timepoints, voxel_features).
    data_4d shape: (X, Y, Z, T)
    mask shape:    (X, Y, Z)

    Result: X shape = (T, num_masked_voxels)
    """
    if data_4d.shape[:3] != mask.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match data spatial shape {data_4d.shape[:3]}"
        )

    X = data_4d[mask].T  # masked voxels become features, transpose to (T, V)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {X.shape}")

    return X


def build_pipeline(pca_components, C, kernel, gamma):
    """
    Build ML pipeline:
    StandardScaler -> optional PCA -> SVM
    """
    steps = [("scaler", StandardScaler())]

    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components, random_state=RANDOM_STATE)))

    if kernel == "linear":
        clf = SVC(C=C, kernel=kernel)
    else:
        clf = SVC(C=C, kernel=kernel, gamma=gamma)

    steps.append(("svm", clf))
    return Pipeline(steps)


def evaluate_one_setting(X, y, k, pca_components, C, kernel, gamma):
    """
    Evaluate one parameter combination using stratified K-fold CV.
    Returns mean CV accuracy.
    """
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    pipeline = build_pipeline(pca_components=pca_components, C=C, kernel=kernel, gamma=gamma)

    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=None)
    return float(np.mean(scores))


def run_grid_search_for_dataset(dataset_name: str, nifti_path: Path, labels: np.ndarray, output_dir: Path):
    """
    For one dataset:
    1) load nifti
    2) search threshold + PCA + SVM params + K
    3) save best mask
    4) save results
    """
    print("=" * 80)
    print(f"Processing dataset: {dataset_name}")
    print(f"NIfTI file: {nifti_path}")

    reference_img, data_4d = load_nifti_4d(nifti_path)
    n_timepoints = data_4d.shape[3]

    if n_timepoints != len(labels):
        raise ValueError(
            f"Number of volumes ({n_timepoints}) does not match number of labels ({len(labels)}) "
            f"for dataset {dataset_name}"
        )

    results = []
    best_result = None
    best_score = -np.inf
    best_mask = None

    for threshold_ratio in THRESHOLDS:
        try:
            mask = create_brain_mask(data_4d, threshold_ratio)
            X = extract_masked_features(data_4d, mask)
        except Exception as e:
            print(f"[Skip] threshold={threshold_ratio} failed: {e}")
            continue

        n_features = X.shape[1]
        valid_pca_list = []

        for pca_comp in PCA_COMPONENTS:
            if pca_comp is None:
                valid_pca_list.append(None)
            else:
                # PCA components must be <= min(n_samples, n_features)
                if pca_comp <= min(X.shape[0], n_features):
                    valid_pca_list.append(pca_comp)

        for k, pca_comp, C, kernel, gamma in product(
            K_VALUES, valid_pca_list, C_VALUES, KERNELS, GAMMA_VALUES
        ):
            # gamma only matters for rbf. Keep one effective setting for linear to avoid duplicate work.
            if kernel == "linear" and gamma != "scale":
                continue

            try:
                mean_acc = evaluate_one_setting(
                    X=X,
                    y=labels,
                    k=k,
                    pca_components=pca_comp,
                    C=C,
                    kernel=kernel,
                    gamma=gamma,
                )

                record = {
                    "dataset": dataset_name,
                    "nifti_path": str(nifti_path),
                    "threshold_ratio": threshold_ratio,
                    "num_masked_voxels": int(np.sum(mask)),
                    "num_features_after_mask": int(n_features),
                    "k_folds": k,
                    "pca_components": pca_comp,
                    "svm_C": C,
                    "svm_kernel": kernel,
                    "svm_gamma": gamma if kernel == "rbf" else None,
                    "mean_cv_accuracy": mean_acc,
                }
                results.append(record)

                print(
                    f"[{dataset_name}] threshold={threshold_ratio:.2f}, "
                    f"voxels={np.sum(mask)}, K={k}, PCA={pca_comp}, "
                    f"C={C}, kernel={kernel}, gamma={gamma if kernel == 'rbf' else 'N/A'} "
                    f"-> mean CV acc={mean_acc:.4f}"
                )

                if mean_acc > best_score:
                    best_score = mean_acc
                    best_result = record
                    best_mask = mask.copy()

            except Exception as e:
                print(
                    f"[Skip] dataset={dataset_name}, threshold={threshold_ratio}, K={k}, "
                    f"PCA={pca_comp}, C={C}, kernel={kernel}, gamma={gamma} failed: {e}"
                )

    if best_result is None or best_mask is None:
        raise RuntimeError(f"No valid parameter combination succeeded for dataset {dataset_name}")

    # Save best mask
    mask_path = output_dir / f"{dataset_name}_best_mask.nii.gz"
    save_mask_nifti(best_mask, reference_img, mask_path)

    # Save all results
    results_path = output_dir / f"{dataset_name}_all_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save best result
    best_result["best_mask_path"] = str(mask_path)
    best_result_path = output_dir / f"{dataset_name}_best_result.json"
    with open(best_result_path, "w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=2)

    print("\nBest result:")
    print(json.dumps(best_result, indent=2))
    print(f"Best mask saved to: {mask_path}")
    print(f"All results saved to: {results_path}")
    print(f"Best result saved to: {best_result_path}")
    print("=" * 80)

    return best_result, results


def save_summary_report(summary_dict: dict, output_dir: Path):
    """
    Save a concise text summary that can help fill the required README.
    """
    summary_path = output_dir / "summary_for_readme.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Assignment 3 Summary\n")
        f.write("=" * 80 + "\n\n")

        for dataset_name, result in summary_dict.items():
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Best mean CV accuracy: {result['mean_cv_accuracy']:.4f}\n")
            f.write(f"Threshold ratio: {result['threshold_ratio']}\n")
            f.write(f"Masked voxels: {result['num_masked_voxels']}\n")
            f.write(f"K folds: {result['k_folds']}\n")
            f.write(f"PCA components: {result['pca_components']}\n")
            f.write(f"SVM C: {result['svm_C']}\n")
            f.write(f"SVM kernel: {result['svm_kernel']}\n")
            f.write(f"SVM gamma: {result['svm_gamma']}\n")
            f.write(f"Mask path: {result['best_mask_path']}\n")
            f.write("\n")

    print(f"Summary for README saved to: {summary_path}")


def main():
    ensure_output_dir(OUTPUT_DIR)

    print("Loading labels...")
    labels = load_labels(LABEL_PATH)
    print(f"Labels loaded. Shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")

    print("\nFinding NIfTI files...")
    dataset_paths = find_nifti_files(DATA_DIR)
    for dataset_name, path in dataset_paths.items():
        print(f"{dataset_name}: {path}")

    summary_dict = {}

    for dataset_name, nifti_path in dataset_paths.items():
        best_result, _ = run_grid_search_for_dataset(
            dataset_name=dataset_name,
            nifti_path=nifti_path,
            labels=labels,
            output_dir=OUTPUT_DIR,
        )
        summary_dict[dataset_name] = best_result

    save_summary_report(summary_dict, OUTPUT_DIR)

    print("\nDone. Both datasets were processed independently as required.")
    print("You can now use the saved best results to fill UNI_Name_Assignment3.README.")


if __name__ == "__main__":
    main()
