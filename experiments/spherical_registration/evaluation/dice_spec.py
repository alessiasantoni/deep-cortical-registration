"""
Dice (global + per-region) of spheres in NET_DIR against FreeSurfer.

Output CSV: dataset, subject, global_dice, dice_precentral, dice_postcentral
"""

from pathlib import Path
import csv
import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

HEMI = "lh"
FSAVERAGE_SPHERE = Path("experiments/spherical_registration/fsaverage_aparc/lh.fsaverage.sphere")
FSAVERAGE_ANNOT  = Path("experiments/spherical_registration/fsaverage_aparc/lh.aparc.annot")
DATA_ROOT = Path("/projects/brainnet-data/sphere-reg")

NET_DIR = Path("/home/s252125/my_project/experiments/sphere_reg2/runs/precomputed_rot/precomp_test_FIXED2")

OUTLIER_CSV = Path(
    "/home/s252125/my_project/experiments/sphere_reg2/runs/precomputed_rot/parcel_dice_FIXED2/outlier_subjects_mine.csv"
)

REGIONS_OF_INTEREST = ["precentral", "postcentral"]

OUT_CSV = Path(
    "/home/s252125/my_project/experiments/sphere_reg2/runs/precomputed_rot/parcel_dice_FIXED2/dice_outliers_only.csv"
)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def normalize_to_unit_sphere(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def transfer_labels(source_sphere, source_labels, target_sphere):
    src = normalize_to_unit_sphere(source_sphere)
    tgt = normalize_to_unit_sphere(target_sphere)
    tree = cKDTree(src)
    _, nn = tree.query(tgt)
    return source_labels[nn]


def dice_for_region(labels_a, labels_b, region_names, region_name):
    lid = None
    for i, name in enumerate(region_names):
        nm = name.decode() if isinstance(name, bytes) else name
        if nm == region_name:
            lid = i
            break
    if lid is None:
        return None
    a, b = labels_a == lid, labels_b == lid
    denom = a.sum() + b.sum()
    if denom == 0:
        return None
    inter = np.logical_and(a, b).sum()
    return float(2.0 * inter / denom)


def global_dice(labels_a, labels_b):
    num, den = 0, 0
    for lid in np.union1d(np.unique(labels_a), np.unique(labels_b)):
        if lid <= 0:
            continue
        a, b = labels_a == lid, labels_b == lid
        num += 2 * np.logical_and(a, b).sum()
        den += a.sum() + b.sum()
    return float(num / den) if den > 0 else np.nan


def compute_dice_metrics(verts, labels_fs, fsavg_sphere, fsavg_labels, region_names):
    labels_pred = transfer_labels(fsavg_sphere, fsavg_labels, verts)
    out = {"global_dice": global_dice(labels_pred, labels_fs)}
    for region in REGIONS_OF_INTEREST:
        out[f"dice_{region}"] = dice_for_region(labels_pred, labels_fs, region_names, region)
    return out


def load_outlier_list(outlier_csv):
    """Reads (dataset, subject) from OUTLIER_CSV.

    Expects at least the 'dataset' and 'subject' columns. Falls back to a
    single column like 'ABIDE_sub-XXXX' if needed.
    """
    with open(outlier_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Columns found in {outlier_csv.name}: {fieldnames}")

    if "dataset" in fieldnames and "subject" in fieldnames:
        return [(r["dataset"].strip(), r["subject"].strip()) for r in rows]

    id_col = None
    for cand in ("subject_id", "id", "name"):
        if cand in fieldnames:
            id_col = cand
            break
    if id_col is not None:
        out = []
        for r in rows:
            dataset, _, subject = r[id_col].partition("_")
            out.append((dataset.strip(), subject.strip()))
        return out

    raise ValueError(
        f"Cannot find 'dataset'/'subject' columns in {outlier_csv}. "
        f"Available columns: {fieldnames}."
    )


def main():
    fsavg_sphere, _ = nib.freesurfer.read_geometry(str(FSAVERAGE_SPHERE))
    fsavg_labels, _, region_names = nib.freesurfer.read_annot(str(FSAVERAGE_ANNOT))

    outlier_list = load_outlier_list(OUTLIER_CSV)
    if not outlier_list:
        print(f"No subjects found in {OUTLIER_CSV}.")
        return
    print(f"\n{len(outlier_list)} outlier subjects to evaluate\n")

    rows = []
    for dataset, subject in outlier_list:
        pred_path = NET_DIR / f"{dataset}_{subject}_{HEMI}.sphere.reg"
        fs_path = DATA_ROOT / dataset / subject / "derivatives" / f"{HEMI}.sphere.reg.freesurfer"

        if not pred_path.exists():
            print(f"  [{dataset}/{subject}] missing sphere in {NET_DIR}, skip")
            continue
        if not fs_path.exists():
            print(f"  [{dataset}/{subject}] missing FS, skip")
            continue

        pred_verts, _ = nib.freesurfer.read_geometry(str(pred_path))
        fs_verts, _ = nib.freesurfer.read_geometry(str(fs_path))
        labels_fs = transfer_labels(fsavg_sphere, fsavg_labels, fs_verts)

        m = compute_dice_metrics(pred_verts, labels_fs, fsavg_sphere, fsavg_labels, region_names)

        row = dict(dataset=dataset, subject=subject, **m)
        rows.append(row)

        print(f"[{dataset}/{subject}] global={m['global_dice']:.4f}  "
              f"precentral={m['dice_precentral']:.4f}  postcentral={m['dice_postcentral']:.4f}")

    if not rows:
        print("\nNo rows produced -- check the paths.")
        return

    fieldnames = ["dataset", "subject", "global_dice", "dice_precentral", "dice_postcentral"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV -> {OUT_CSV}")


if __name__ == "__main__":
    main()