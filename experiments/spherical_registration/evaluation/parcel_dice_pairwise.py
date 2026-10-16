"""
Pairwise comparison, on real test subjects
(2-3 datasets of choice): Dice (shared fsaverage atlas, transferred via
each registration), MAE and correlation (subject's sulc, already on
canonical indexing -- no resampling needed).
Reuses the spheres ALREADY predicted by the network (no new inference).

Usage (from the repository root, after inference.py):
  python -m experiments.spherical_registration.evaluation.parcel_dice_pairwise --run_dir runs/ico6_train
Reads <run_dir>/test_predictions, writes <run_dir>/dice_pairwise/.

python -m experiments.spherical_registration.evaluation.parcel_dice_pairwise --run_dir runs/ico6_train

"""

from pathlib import Path
import argparse
import itertools
import json

import numpy as np
import torch
import nibabel as nib
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon, pearsonr
import brainsynth.resources

# ============== CONFIG ==============
HEMI = "lh"
DATA_ROOT = Path("/projects/brainnet-data/sphere-reg")
SPLIT_DIR = Path("/projects/brainnet-data/subject_splits")
BAD_SURFACES = SPLIT_DIR / "bad_surfaces.json"

# fsaverage aparc resampled on the brainsynth template mesh: its labels are
# indexed like the vertices of the template sphere.reg, so it must be paired with that sphere
FSAVERAGE_ANNOT = Path(f"experiments/spherical_registration/fsaverage_aparc/{HEMI}.aparc.annot")

# --- Choose here the 2-3 datasets you want to test ---
DATASETS = ["OASIS3", "HCP"]
N_MAX_PER_DATASET = 30

PRED_SPHERES_DIR = None   # set in main() from --run_dir
# ====================================


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def normalize_to_unit_sphere(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def transfer_scalar(source_sphere, source_values, target_sphere):
    """Generic nearest-neighbor, for labels OR continuous values."""
    src = normalize_to_unit_sphere(source_sphere)
    tgt = normalize_to_unit_sphere(target_sphere)
    tree = cKDTree(src)
    _, nn = tree.query(tgt)
    return source_values[nn]


def global_dice(labels_a, labels_b):
    num, den = 0, 0
    for lid in np.union1d(np.unique(labels_a), np.unique(labels_b)):
        if lid <= 0:
            continue
        a, b = labels_a == lid, labels_b == lid
        num += 2 * np.logical_and(a, b).sum()
        den += a.sum() + b.sum()
    return float(num / den) if den > 0 else np.nan


def read_split_subjects(dataset_name, split, split_dir, bad_surfaces_file):
    split_file = Path(split_dir) / f"{dataset_name}.{split}.txt"
    if not split_file.exists():
        return []
    with open(split_file) as f:
        subject_ids = [line.strip() for line in f if line.strip()]
    bad = {}
    if bad_surfaces_file is not None and Path(bad_surfaces_file).exists():
        with open(bad_surfaces_file) as f:
            bad = json.load(f)
    bad_set = set(bad.get(dataset_name, []))
    return [s for s in subject_ids if s not in bad_set]


def process_subject(dataset, subject, fsavg_sphere, fsavg_labels):
    """No native resampling needed: sulc and spheres already share the
    same canonical indexing by construction (Nielsen's method)."""
    mine_sphere_path = PRED_SPHERES_DIR / f"{dataset}_{subject}_{HEMI}.sphere.reg"
    fs_sphere_path = DATA_ROOT / dataset / subject / "derivatives" / f"{HEMI}.sphere.reg.freesurfer"
    sulc_path = DATA_ROOT / dataset / subject / "derivatives" / f"{HEMI}.sulc.pt"

    if not all(p.exists() for p in [mine_sphere_path, fs_sphere_path, sulc_path]):
        return None

    mine_verts, _ = nib.freesurfer.read_geometry(str(mine_sphere_path))
    fs_verts, _ = nib.freesurfer.read_geometry(str(fs_sphere_path))
    sulc = torch.load(sulc_path, weights_only=True).numpy().astype(np.float32)

    n = mine_verts.shape[0]
    if not (fs_verts.shape[0] == n and sulc.shape[0] == n):
        print(f"  [{dataset}/{subject}] vertex MISMATCH, skip")
        return None

    sulc_z = normalize(sulc)

    # --- Dice: SHARED fsaverage atlas, transferred via each registration ---
    label_mine = transfer_scalar(fsavg_sphere, fsavg_labels, mine_verts)
    label_fs = transfer_scalar(fsavg_sphere, fsavg_labels, fs_verts)

    # --- MAE/correlation: SUBJECT's sulc, transferred via each registration ---
    sulc_mine_common = transfer_scalar(mine_verts, sulc_z, fsavg_sphere)
    sulc_fs_common = transfer_scalar(fs_verts, sulc_z, fsavg_sphere)

    return dict(
        label_mine=label_mine.astype(np.int32), sulc_mine=sulc_mine_common,
        label_fs=label_fs.astype(np.int32), sulc_fs=sulc_fs_common,
    )


def main():
    global PRED_SPHERES_DIR
    parser = argparse.ArgumentParser(description="Pairwise Dice / MAE / correlation, network vs FreeSurfer.")
    parser.add_argument("--run_dir", default="runs/ico6_train",
                        help="folder containing test_predictions/ (written by inference.py)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    PRED_SPHERES_DIR = run_dir / "test_predictions"
    out_dir = run_dir / "dice_pairwise"
    out_dir.mkdir(parents=True, exist_ok=True)

    fsavg_sphere = brainsynth.resources.Template().load_surface(HEMI, "sphere.reg")["vertices"].numpy()
    fsavg_labels, _, _ = nib.freesurfer.read_annot(str(FSAVERAGE_ANNOT))
    assert fsavg_sphere.shape[0] == fsavg_labels.shape[0], "atlas and template sphere do not match"

    all_subjects = []
    for dataset in DATASETS:
        subs = read_split_subjects(dataset, "test", SPLIT_DIR, BAD_SURFACES)
        if N_MAX_PER_DATASET is not None:
            subs = subs[:N_MAX_PER_DATASET]
        print(f"  {dataset}.test: {len(subs)} subjects (limited)")
        for sub in subs:
            all_subjects.append((dataset, sub))
    print(f"\nTOTAL: {len(all_subjects)} subjects from {len(DATASETS)} datasets\n")

    data = {}
    for dataset, subject in all_subjects:
        result = process_subject(dataset, subject, fsavg_sphere, fsavg_labels)
        if result is None:
            continue
        data[(dataset, subject)] = result
    print(f"Valid subjects: {len(data)}\n")

    valid_keys = list(data.keys())
    rows = {"dice_mine": [], "dice_fs": [], "mae_mine": [], "mae_fs": [],
            "corr_mine": [], "corr_fs": []}

    for key_a, key_b in itertools.combinations(valid_keys, 2):
        da, db = data[key_a], data[key_b]
        rows["dice_mine"].append(global_dice(da["label_mine"], db["label_mine"]))
        rows["dice_fs"].append(global_dice(da["label_fs"], db["label_fs"]))
        rows["mae_mine"].append(float(np.mean(np.abs(da["sulc_mine"] - db["sulc_mine"]))))
        rows["mae_fs"].append(float(np.mean(np.abs(da["sulc_fs"] - db["sulc_fs"]))))
        r_mine, _ = pearsonr(da["sulc_mine"], db["sulc_mine"])
        r_fs, _ = pearsonr(da["sulc_fs"], db["sulc_fs"])
        rows["corr_mine"].append(r_mine)
        rows["corr_fs"].append(r_fs)

    n_pairs = len(rows["dice_mine"])
    print(f"=== Summary over {n_pairs} pairs ({len(valid_keys)} subjects, {DATASETS}) ===\n")

    for metric_name, higher_is_better in [("dice", True), ("mae", False), ("corr", True)]:
        arr_mine = np.array(rows[f"{metric_name}_mine"])
        arr_fs = np.array(rows[f"{metric_name}_fs"])
        stat, p = wilcoxon(arr_mine, arr_fs)
        diff = arr_mine - arr_fs
        n_wins = (diff > 0).sum() if higher_is_better else (diff < 0).sum()
        print(f"  --- {metric_name.upper()} ---")
        print(f"    My network: {arr_mine.mean():.4f} +/- {arr_mine.std():.4f}")
        print(f"    FreeSurfer: {arr_fs.mean():.4f} +/- {arr_fs.std():.4f}")
        print(f"    Wilcoxon p={p:.6f}   ({n_wins}/{n_pairs} pairs in my favor)\n")

    np.savez(out_dir / "s3reg_style_real_data.npz", **{k: np.array(v) for k, v in rows.items()})
    print(f"Saved -> {out_dir / 's3reg_style_real_data.npz'}")


if __name__ == "__main__":
    main()
