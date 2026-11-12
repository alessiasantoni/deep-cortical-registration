"""
How far the region boundaries are from FreeSurfer's, in mm: HD90 (90th percentile
of the boundary distances), HD100 (max) and MBD (mean).

The aparc is moved onto the subject with each registration (no registration, JOSA,
ours) and with FreeSurfer, as in parcel_dice.py. For each region, its boundary
vertices are the ones with a neighbour of another region. Distances are measured on
the subject's lh.white, from each boundary vertex of one parcellation to the closest
one of the other, in both directions. If a region is missing in one of the two the
value is NaN. "global" means all regions pooled together.

Subjects are those of <run_dir>/parcel_dice/dice_mine_vs_fs.csv, so run parcel_dice.py
first (with the same --pred_dir). Outputs, in <run_dir>/parcel_dice:
  hausdorff_<method>_vs_fs.csv      one row per subject
  hausdorff_per_region_summary.csv  median HD90 per region

  python -m experiments.spherical_registration.evaluation.hausdorff_boundaries --run_dir runs/ico6_train
"""

from pathlib import Path
from multiprocessing import Pool
import argparse
import csv

import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

import brainsynth.resources

from experiments.spherical_registration.evaluation.atlas import add_fsaverage_argument, load_fsaverage_atlas
from experiments.spherical_registration import finetune_train

DATA_ROOT = Path(finetune_train.DATA_ROOT)

HEMI = "lh"
PRED_DIR = None      # set in main()
N_VERTS = 245762
PERCENTILE = 90
N_WORKERS = 16
METHODS = ["initial", "josa", "mine"]


def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# filled once in init_globals(), the worker processes get them through fork
FSAVG_TREE = None
FSAVG_LABELS = None
REGION_IDS = None
LABELS_INITIAL = None


def init_globals(fsaverage_dir):
    global FSAVG_TREE, FSAVG_LABELS, REGION_IDS, LABELS_INITIAL
    fsavg, FSAVG_LABELS, names = load_fsaverage_atlas(fsaverage_dir, HEMI)
    FSAVG_TREE = cKDTree(unit(fsavg))
    REGION_IDS = {names[i].decode(): i for i in np.unique(FSAVG_LABELS) if i > 0}
    template = brainsynth.resources.Template().load_surface(HEMI, "sphere.reg")["vertices"].numpy()
    LABELS_INITIAL = transfer(template[:N_VERTS])


def transfer(sphere):
    """fsaverage label of the nearest fsaverage vertex, for every vertex of `sphere`."""
    _, nn = FSAVG_TREE.query(unit(sphere))
    return FSAVG_LABELS[nn]


def boundary_mask(labels, edges):
    """True for vertices with at least one neighbour carrying a different label."""
    diff = labels[edges[:, 0]] != labels[edges[:, 1]]
    mask = np.zeros(len(labels), bool)
    mask[edges[diff, 0]] = True
    mask[edges[diff, 1]] = True
    return mask


def region_distances(coords, lab_a, bnd_a, lab_b, bnd_b, rid):
    pa = coords[bnd_a & (lab_a == rid)]
    pb = coords[bnd_b & (lab_b == rid)]
    if len(pa) == 0 or len(pb) == 0:
        return None
    d_ab, _ = cKDTree(pb).query(pa)
    d_ba, _ = cKDTree(pa).query(pb)
    return np.concatenate([d_ab, d_ba])


def process(subject_key):
    dataset, subject = subject_key
    sub_dir = DATA_ROOT / dataset / subject
    white, faces = nib.freesurfer.read_geometry(str(sub_dir / f"{HEMI}.white"))
    if white.shape[0] != N_VERTS:
        return None
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])

    spheres = {
        "fs": nib.freesurfer.read_geometry(str(sub_dir / "derivatives" / f"{HEMI}.sphere.reg.freesurfer"))[0],
        "josa": nib.freesurfer.read_geometry(str(sub_dir / "derivatives" / f"{HEMI}.sphere.reg.josa"))[0],
        "mine": nib.freesurfer.read_geometry(str(PRED_DIR / f"{dataset}_{subject}_{HEMI}.sphere.reg"))[0],
    }
    labels = {m: transfer(s) for m, s in spheres.items()}
    labels["initial"] = LABELS_INITIAL
    bnd = {m: boundary_mask(l, edges) for m, l in labels.items()}

    out = {}
    for m in METHODS:
        per_region, pooled = {}, []
        for name, rid in REGION_IDS.items():
            d = region_distances(white, labels[m], bnd[m], labels["fs"], bnd["fs"], rid)
            if d is None:
                per_region[name] = (np.nan, np.nan, np.nan)
                continue
            per_region[name] = (np.percentile(d, PERCENTILE), d.max(), d.mean())
            pooled.append(d)
        pooled = np.concatenate(pooled)
        out[m] = dict(
            regions={k: v[0] for k, v in per_region.items()},
            hd90_mean_regions=np.nanmean([v[0] for v in per_region.values()]),
            hd90_global=np.percentile(pooled, PERCENTILE),
            hd100_global=pooled.max(),
            mbd_global=pooled.mean(),
        )
    return dataset, subject, out


def main():
    global PRED_DIR, N_WORKERS
    parser = argparse.ArgumentParser(description="Boundary Hausdorff distances (HD90, HD100, MBD) vs FreeSurfer.")
    parser.add_argument("--run_dir", default="runs/ico6_train",
                        help="run folder containing parcel_dice/ (written by parcel_dice.py)")
    parser.add_argument("--pred_dir", default=None,
                        help="folder with the predicted spheres (default: <run_dir>/test_predictions)")
    parser.add_argument("--workers", type=int, default=N_WORKERS, help="parallel processes")
    add_fsaverage_argument(parser)
    args = parser.parse_args()

    PRED_DIR = Path(args.pred_dir) if args.pred_dir else Path(args.run_dir) / "test_predictions"
    RESULTS_DIR = Path(args.run_dir) / "parcel_dice"
    N_WORKERS = args.workers

    with open(RESULTS_DIR / "dice_mine_vs_fs.csv", newline="") as f:
        subjects = [(r[0], r[1]) for r in list(csv.reader(f))[1:]]
    print(f"{len(subjects)} subjects from {RESULTS_DIR / 'dice_mine_vs_fs.csv'}")

    init_globals(args.fsaverage_dir)
    regions = sorted(REGION_IDS)
    with Pool(N_WORKERS) as pool:
        results = [r for r in pool.imap(process, subjects, chunksize=4) if r is not None]
    print(f"Evaluated: {len(results)} / {len(subjects)}")

    summary_cols = ["hd90_mean_regions", "hd90_global", "hd100_global", "mbd_global"]
    for m in METHODS:
        path = RESULTS_DIR / f"hausdorff_{m}_vs_fs.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "subject"] + [f"hd90_{r}" for r in regions] + summary_cols)
            for ds, sub, out in results:
                o = out[m]
                w.writerow([ds, sub] + [o["regions"][r] for r in regions] + [o[c] for c in summary_cols])
        arr = np.array([[out[m][c] for c in summary_cols] for _, _, out in results])
        print(f"\n{m}: " + "  ".join(f"{c} {np.median(arr[:, i]):.2f} mm (median)"
                                      for i, c in enumerate(summary_cols)))

    with open(RESULTS_DIR / "hausdorff_per_region_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region"] + [f"hd90_median_{m}" for m in METHODS] + ["mine_minus_josa"])
        for r in regions:
            med = [np.nanmedian([out[m]["regions"][r] for _, _, out in results]) for m in METHODS]
            w.writerow([r] + med + [med[2] - med[1]])
    print(f"\nSaved hausdorff_*_vs_fs.csv and hausdorff_per_region_summary.csv in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
