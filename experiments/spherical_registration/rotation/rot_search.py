"""
Precomputes, for EVERY train+validation subject (not just outliers), the
rigid rotation that minimizes the MSE on sulc, restricted to +-20 degrees
per axis (same range as the previous augmentation -- so the network is
trained on realistic corrections, never outside the range it will see later).

Search at ico3 resolution (3842 vertices) -- the same one used by the
coarse stage in training, no need for higher resolution here.

Incremental saving (resume-safe): if the job is interrupted, relaunch it
and it resumes from where it left off.
"""

from pathlib import Path
import csv
import json

import numpy as np
import torch
import nibabel as nib
from scipy.spatial.transform import Rotation as R

import brainsynth.resources
from experiments.sphere_reg2.dataset import sample_grid_at_vertices

# ============== CONFIG ==============
DATA_ROOT = Path("/projects/brainnet-data/sphere-reg")
SPLIT_DIR = Path("/projects/brainnet-data/subject_splits")
BAD_SURFACES = SPLIT_DIR / "bad_surfaces.json"
HEMI = "lh"

# Training & validation
#DATASETS = ["ABIDE", "ADHD200", "ADNI3", "AIBL", "Buckner40","Chinese-HCP", "HCP", "OASIS3", "ISBI2015"]
#SPLITS = ["train", "validation"]
# Test
DATASETS = ["ABIDE", "ADHD200", "ADNI3", "AIBL", "Buckner40","Chinese-HCP", "HCP", "OASIS3", "ISBI2015", "MCIC", "COBRE"] 
SPLITS = ["test"]

GRID_PATH = "/home/s252125/my_project/experiments/sphere_reg2/fsaverage_grid/lh.white.features.npz"
TEMPLATE_SULC_STATS = {
    "lh": {"mean": -1.1523885e-08, "std": 0.59384775},
    "rh": {"mean": -4.1938106e-09, "std": 0.5924748},
}
SPHERE_RADIUS = 100.0
SEARCH_N_VERTS = 3842   # ico3 -- same resolution as the coarse stage

OUT_DIR = Path(
    "/home/s252125/my_project/experiments/sphere_reg2/runs/precomputed_rotations_testset"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "rotation_lookup.csv"

# --- Range RESTRICTED to +-20 degrees, coarse-to-fine within that budget ---
MAX_RANGE_DEG = 20.0
COARSE_STEP_DEG = 5.0
FINE_RANGE_DEG = 6.0
FINE_STEP_DEG = 1.0
# ====================================


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def compute_mse(a, b):
    a_n, b_n = zscore(a), zscore(b)
    return float(((a_n - b_n) ** 2).mean())


def standardize_template_grid(template_grid, hemi):
    stats = TEMPLATE_SULC_STATS[hemi]
    return (template_grid - stats["mean"]) / stats["std"]


def load_template_grid(grid_path, hemi):
    grid_data = np.load(grid_path)
    return torch.tensor(grid_data["sulc"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def get_template_sphere_init_np(hemi, sphere_radius):
    template_resources = brainsynth.resources.Template()
    tmpl = template_resources.load_surface(hemi, "sphere.reg")["vertices"].numpy()
    tmpl = tmpl / np.linalg.norm(tmpl, axis=1, keepdims=True) * sphere_radius
    return tmpl.astype(np.float32)


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


def grid_search_rotation_sulc_mse(sulc_subj_np, template_grid_torch, base_verts_np,
                                   center_deg, half_range_deg, step_deg):
    ca, cb, cg = center_deg
    a_vals = np.arange(ca - half_range_deg, ca + half_range_deg + 1e-6, step_deg)
    b_vals = np.arange(cb - half_range_deg, cb + half_range_deg + 1e-6, step_deg)
    g_vals = np.arange(cg - half_range_deg, cg + half_range_deg + 1e-6, step_deg)

    best_mse = np.inf
    best_angles = (0.0, 0.0, 0.0)

    for a in a_vals:
        for b in b_vals:
            for g in g_vals:
                rot = R.from_euler("ZYZ", [a, b, g], degrees=True)
                rotated_verts = rot.apply(base_verts_np).astype(np.float32)
                v_t = torch.from_numpy(rotated_verts.T).unsqueeze(0).float()
                sulc_sampled = sample_grid_at_vertices(template_grid_torch, v_t)[0, 0].numpy()
                mse = compute_mse(sulc_subj_np, sulc_sampled)
                if mse < best_mse:
                    best_mse = mse
                    best_angles = (a, b, g)

    return best_angles, best_mse


def already_processed(log_path):
    done = set()
    if log_path.exists():
        with open(log_path) as f:
            for row in csv.DictReader(f):
                done.add((row["dataset"], row["subject"], row["split"]))
    return done


def main():
    print("Loading template grid ...")
    template_grid = load_template_grid(GRID_PATH, HEMI)
    template_grid = standardize_template_grid(template_grid, HEMI)

    base_verts_full = get_template_sphere_init_np(HEMI, SPHERE_RADIUS)
    base_verts_search = base_verts_full[:SEARCH_N_VERTS]

    all_subjects = []
    for split in SPLITS:
        for dataset in DATASETS:
            subs = read_split_subjects(dataset, split, SPLIT_DIR, BAD_SURFACES)
            print(f"  {dataset}.{split}: {len(subs)} subjects")
            for sub in subs:
                all_subjects.append((dataset, sub, split))
    print(f"\nTOTAL: {len(all_subjects)} subjects to process (+-{MAX_RANGE_DEG} degrees)\n")

    done = already_processed(LOG_PATH)
    if done:
        print(f"{len(done)} already processed in the existing log -> will be skipped\n")

    write_header = not LOG_PATH.exists()
    log_file = open(LOG_PATH, "a", newline="")
    writer = csv.writer(log_file)
    if write_header:
        writer.writerow(["dataset", "subject", "split", "mse_no_rotation",
                          "mse_with_rotation", "alpha", "beta", "gamma"])

    n_processed = 0
    for i, (dataset, subject, split) in enumerate(all_subjects, 1):
        if (dataset, subject, split) in done:
            continue

        sulc_path = DATA_ROOT / dataset / subject / "derivatives" / f"{HEMI}.sulc"
        if not sulc_path.exists():
            continue

        sulc_subj_full = nib.freesurfer.read_morph_data(str(sulc_path))
        sulc_subj_search = sulc_subj_full[:SEARCH_N_VERTS].astype(np.float32)

        v0_t = torch.from_numpy(base_verts_search.T).unsqueeze(0).float()
        sulc0 = sample_grid_at_vertices(template_grid, v0_t)[0, 0].numpy()
        mse_no_rotation = compute_mse(sulc_subj_search, sulc0)

        coarse_angles, _ = grid_search_rotation_sulc_mse(
            sulc_subj_search, template_grid, base_verts_search,
            center_deg=(0.0, 0.0, 0.0), half_range_deg=MAX_RANGE_DEG, step_deg=COARSE_STEP_DEG,
        )
        fine_angles, fine_mse = grid_search_rotation_sulc_mse(
            sulc_subj_search, template_grid, base_verts_search,
            center_deg=coarse_angles, half_range_deg=FINE_RANGE_DEG, step_deg=FINE_STEP_DEG,
        )

        writer.writerow([dataset, subject, split, mse_no_rotation, fine_mse,
                          round(fine_angles[0], 3), round(fine_angles[1], 3), round(fine_angles[2], 3)])
        log_file.flush()

        n_processed += 1
        if n_processed % 50 == 0 or i == len(all_subjects):
            print(f"  [{i}/{len(all_subjects)}] {dataset}/{subject} ({split})  "
                  f"mse {mse_no_rotation:.4f}->{fine_mse:.4f}  angles={tuple(round(a,1) for a in fine_angles)}")

    log_file.close()
    print(f"\nDone. {n_processed} new subjects processed.")
    print(f"Lookup table -> {LOG_PATH}")


if __name__ == "__main__":
    main()