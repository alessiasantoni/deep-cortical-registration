"""
Boundary distances (HD90, HD100, MBD, in mm) against the manual labels of
Mindboggle-101. Same computation as evaluation/hausdorff_boundaries.py, but the
reference are the manual DKT31 labels instead of FreeSurfer, and the labels of each
method are built as in dice_vs_manual.py. Distances are measured on lh.white, which
has the same vertices as the manual annot and the spheres. Lower is better.

  python -m experiments.spherical_registration.evaluation.manual_labels.hausdorff_vs_manual \
      --data_root <Mindboggle101> --id_map <id_map.txt> --run_dir runs/ico6_train

Writes <run_dir>/hausdorff_vs_manual/ (or --out_dir).
"""

import argparse
from pathlib import Path
from multiprocessing import Pool
import csv

import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

import brainsynth.resources

from experiments.spherical_registration.evaluation.manual_labels.common import (
    HEMI, add_data_arguments, add_fsaverage_argument, fsaverage_files, load_subjects,
)
from experiments.spherical_registration.evaluation.manual_labels.dkt_labels import (
    DKT31_REGIONS, aparc_atlas_to_dkt31, read_annot_names, to_dkt31,
)


# arguments and atlas are read at import, so the worker processes find them already set
parser = argparse.ArgumentParser(description="Boundary distances against the manual Mindboggle-101 labels.")
add_data_arguments(parser)
add_fsaverage_argument(parser)
parser.add_argument("--run_dir", default="runs/ico6_train", help="run folder; results go in <run_dir>/hausdorff_vs_manual")
parser.add_argument("--pred_dir", default=None, help="predicted spheres (default: <run_dir>/mindboggle_predictions)")
parser.add_argument("--out_dir", default=None, help="default: <run_dir>/hausdorff_vs_manual")
parser.add_argument("--workers", type=int, default=16, help="parallel processes")
args = parser.parse_args()

DATA_ROOT = Path(args.data_root)
FSAVERAGE_SPHERE, FSAVERAGE_ANNOT = fsaverage_files(args.fsaverage_dir)
PRED_DIR = Path(args.pred_dir) if args.pred_dir else Path(args.run_dir) / "mindboggle_predictions"
OUTPUT_DIR = Path(args.out_dir) if args.out_dir else Path(args.run_dir) / "hausdorff_vs_manual"

METHODS = {
    "map":        None,
    "josa":       f"derivatives/{HEMI}.sphere.reg.josa",
    "mine":       "PRED",
    "freesurfer": f"derivatives/{HEMI}.sphere.reg.freesurfer",
}
COLORS = {"map": "#eda100", "freesurfer": "#eb6834", "josa": "#1baf7a", "mine": "#2a78d6"}
NAMES = {"map": "No registration", "josa": "JOSA", "mine": "Proposed", "freesurfer": "FreeSurfer"}
PERCENTILE = 90
N_WORKERS = args.workers


def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


fsavg_sphere, _ = nib.freesurfer.read_geometry(str(FSAVERAGE_SPHERE))
ATLAS31 = aparc_atlas_to_dkt31(fsavg_sphere, read_annot_names(FSAVERAGE_ANNOT))
ATLAS_TREE = cKDTree(unit(fsavg_sphere))
TEMPLATE_SPHERE = brainsynth.resources.Template().load_surface(HEMI, "sphere.reg")["vertices"].numpy()


def boundary_mask(labels, edges):
    """True for vertices with at least one neighbour carrying a different label."""
    diff = labels[edges[:, 0]] != labels[edges[:, 1]]
    mask = np.zeros(len(labels), bool)
    mask[edges[diff, 0]] = True
    mask[edges[diff, 1]] = True
    return mask


def region_distances(coords, lab_a, bnd_a, lab_b, bnd_b, region):
    pa = coords[bnd_a & (lab_a == region)]
    pb = coords[bnd_b & (lab_b == region)]
    if len(pa) == 0 or len(pb) == 0:
        return None
    d_ab, _ = cKDTree(pb).query(pa)
    d_ba, _ = cKDTree(pa).query(pb)
    return np.concatenate([d_ab, d_ba])


def process(sub):
    sub_dir = DATA_ROOT / sub
    try:
        manual = to_dkt31(read_annot_names(sub_dir / f"{HEMI}.labels.DKT31.manual.annot"))
    except (IndexError, ValueError, FileNotFoundError):
        return sub, None, "manual DKT31 annot unreadable"
    white, faces = nib.freesurfer.read_geometry(str(sub_dir / f"{HEMI}.white"))
    if white.shape[0] != len(manual):
        return sub, None, f"lh.white has {white.shape[0]} vertices, manual annot {len(manual)}"
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    bnd_manual = boundary_mask(manual, edges)

    out = {}
    for method, rel in METHODS.items():
        path = PRED_DIR / f"{sub}_{HEMI}.sphere.reg" if rel == "PRED" else (sub_dir / rel if rel else None)
        if path is not None and not path.exists():
            return sub, None, f"{path.name} missing"
        sphere = nib.freesurfer.read_geometry(str(path))[0] if path else TEMPLATE_SPHERE
        labels = ATLAS31[ATLAS_TREE.query(unit(sphere))[1]]
        bnd = boundary_mask(labels, edges)

        per_region, pooled = {}, []
        for r in DKT31_REGIONS:
            d = region_distances(white, labels, bnd, manual, bnd_manual, r)
            if d is None:
                per_region[r] = np.nan
                continue
            per_region[r] = np.percentile(d, PERCENTILE)
            pooled.append(d)
        pooled = np.concatenate(pooled)
        out[method] = dict(per_region, hd90_mean_regions=np.nanmean(list(per_region.values())),
                           hd90_global=np.percentile(pooled, PERCENTILE),
                           hd100_global=pooled.max(), mbd_global=pooled.mean())
    return sub, out, None


def main():
    subjects = load_subjects(args.id_map)
    print(f"{len(subjects)} subjects")
    with Pool(N_WORKERS) as pool:
        results = []
        for sub, out, err in pool.imap(process, subjects, chunksize=2):
            if err:
                print(f"  {sub}: {err}")
            else:
                results.append((sub, out))
    n = len(results)
    print(f"{n} subjects evaluated")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_cols = ["hd90_mean_regions", "hd90_global", "hd100_global", "mbd_global"]
    with open(OUTPUT_DIR / "hausdorff_per_subject.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "method"] + [f"hd90_{r}" for r in DKT31_REGIONS] + summary_cols)
        for sub, out in results:
            for m in METHODS:
                w.writerow([sub, m] + [out[m][r] for r in DKT31_REGIONS] + [out[m][c] for c in summary_cols])

    with open(OUTPUT_DIR / "hausdorff_per_region.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region"] + [f"hd90_median_{m}" for m in METHODS])
        for r in DKT31_REGIONS:
            w.writerow([r] + [f"{np.nanmedian([out[m][r] for _, out in results]):.3f}" for m in METHODS])

    def values(m, key):
        return np.array([out[m][key] for _, out in results], float)

    summary = [f"=== Boundary distance vs MANUAL labels (DKT31), Mindboggle-101 {HEMI}, {n} subjects ===",
               "median over subjects [IQR], mm; lower = better\n"]
    for key in summary_cols + ["precentral", "postcentral"]:
        summary.append(f"{key if key in summary_cols else 'hd90_' + key}:")
        for m in METHODS:
            v = values(m, key)
            q1, q3 = np.nanpercentile(v, [25, 75])
            summary.append(f"  {m:<11} {np.nanmedian(v):6.2f}  [{q1:.2f}, {q3:.2f}]")
        mine = values("mine", key)
        for m in METHODS:
            if m == "mine":
                continue
            other = values(m, key)
            ok = ~np.isnan(mine) & ~np.isnan(other)
            diff = mine[ok] - other[ok]
            summary.append(f"  mine - {m:<11} median {np.median(diff):+.2f} mm   mine better (lower) in "
                           f"{int((diff < 0).sum())}/{ok.sum()}   Wilcoxon p = {wilcoxon(mine[ok], other[ok]).pvalue:.1e}")
        summary.append("")
    text = "\n".join(summary)
    (OUTPUT_DIR / "summary.txt").write_text(text)
    print("\n" + text)

    # one point per subject, the bar is the median
    panels = [("hd90_mean_regions", "Mean regional HD90"), ("precentral", "precentral HD90"),
              ("postcentral", "postcentral HD90")]
    plot_methods = list(METHODS)
    names = NAMES
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 4.8))
    rng = np.random.default_rng(0)
    for ax, (key, title) in zip(axes, panels):
        for i, m in enumerate(plot_methods, 1):
            v = values(m, key)
            ax.scatter(rng.normal(i, 0.06, len(v)), v, s=10, color=COLORS[m], alpha=0.6, linewidths=0)
            med = np.nanmedian(v)
            ax.hlines(med, i - 0.3, i + 0.3, color="#0b0b0b", linewidth=2)
            ax.annotate(f"{med:.1f}", (i, med), xytext=(0, 5), textcoords="offset points", ha="center",
                        va="bottom", fontsize=9, fontweight="bold",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
        ax.set_xticks(range(1, len(plot_methods) + 1), [names[m] for m in plot_methods])
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("HD90 vs manual labels (mm), lower = better")
    fig.suptitle(f"Boundary distance vs manual labels, Mindboggle-101 {HEMI} ({n} subjects), bar = median")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"hausdorff_vs_manual.{ext}", dpi=150)
    print(f"results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
