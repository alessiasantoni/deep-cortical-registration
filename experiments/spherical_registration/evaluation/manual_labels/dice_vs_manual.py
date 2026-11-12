"""
Dice against the manual labels of Mindboggle-101.

The fsaverage aparc, converted to DKT31 (dkt_labels.py), is moved onto every
subject with four registrations:
    map         undeformed template (no registration)
    freesurfer  derivatives/lh.sphere.reg.freesurfer
    josa        derivatives/lh.sphere.reg.josa
    mine        <pred_dir>/<sub>_lh.sphere.reg, from inference_mindboggle.py
and compared with the manual labels in the DKT31 (31 regions) and DKT25 (25 regions)
protocols. Vertices without a manual region are ignored. For every subject: Dice per
region, mean_dice (mean over regions) and global_dice (regions pooled, i.e. fraction
of vertices with the right region). Subjects whose manual annot cannot be read are skipped.

  python -m experiments.spherical_registration.evaluation.manual_labels.dice_vs_manual \
      --data_root <Mindboggle101> --id_map <id_map.txt> --run_dir runs/ico6_train

Writes <run_dir>/dice_vs_manual/ (or --out_dir).
"""

import argparse
import csv
from pathlib import Path

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
    DKT25_REGIONS, DKT31_REGIONS,
    aparc_atlas_to_dkt31, read_annot_names, to_dkt25, to_dkt31,
)


parser = argparse.ArgumentParser(description="Parcellation Dice against the manual Mindboggle-101 labels.")
add_data_arguments(parser)
add_fsaverage_argument(parser)
parser.add_argument("--run_dir", default="runs/ico6_train", help="run folder; results go in <run_dir>/dice_vs_manual")
parser.add_argument("--pred_dir", default=None, help="predicted spheres (default: <run_dir>/mindboggle_predictions)")
parser.add_argument("--out_dir", default=None, help="default: <run_dir>/dice_vs_manual")
args = parser.parse_args()

DATA_ROOT = Path(args.data_root)
FSAVERAGE_SPHERE, FSAVERAGE_ANNOT = fsaverage_files(args.fsaverage_dir)
PRED_DIR = Path(args.pred_dir) if args.pred_dir else Path(args.run_dir) / "mindboggle_predictions"
OUTPUT_DIR = Path(args.out_dir) if args.out_dir else Path(args.run_dir) / "dice_vs_manual"

# sphere of each method, relative to the subject folder; None = template, "PRED" = our predictions
METHODS = {
    "map":        None,
    "josa":       f"derivatives/{HEMI}.sphere.reg.josa",
    "mine":       "PRED",
    "freesurfer": f"derivatives/{HEMI}.sphere.reg.freesurfer",
}
PROTOCOLS = {"DKT31": DKT31_REGIONS, "DKT25": DKT25_REGIONS}

COLORS = {"map": "#eda100", "freesurfer": "#eb6834", "josa": "#1baf7a", "mine": "#2a78d6"}


def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def dice_scores(pred, manual, regions):
    """Dice of each region and global Dice (all regions pooled)."""
    per_region = {}
    num = den = 0
    for r in regions:
        a, b = pred == r, manual == r
        d = a.sum() + b.sum()
        inter = np.logical_and(a, b).sum()
        per_region[r] = 2.0 * inter / d if d > 0 else np.nan
        num += 2 * inter
        den += d
    return per_region, float(num / den)


fsavg_sphere, _ = nib.freesurfer.read_geometry(str(FSAVERAGE_SPHERE))
atlas31 = aparc_atlas_to_dkt31(fsavg_sphere, read_annot_names(FSAVERAGE_ANNOT))
atlas_tree = cKDTree(unit(fsavg_sphere))
template_sphere = brainsynth.resources.Template().load_surface(HEMI, "sphere.reg")["vertices"].numpy()


rows = {p: [] for p in PROTOCOLS}
subjects = load_subjects(args.id_map)
print(f"{len(subjects)} subjects")

for sub in subjects:
    sub_dir = DATA_ROOT / sub
    try:
        manual31 = to_dkt31(read_annot_names(sub_dir / f"{HEMI}.labels.DKT31.manual.annot"))
    except (IndexError, ValueError, FileNotFoundError):
        print(f"  {sub}: cannot read the manual annot")
        continue
    # only the vertices that have a manual region count
    mask = manual31 != ""
    manual = {"DKT31": manual31[mask], "DKT25": to_dkt25(manual31)[mask]}

    spheres = {}
    for method, rel in METHODS.items():
        path = PRED_DIR / f"{sub}_{HEMI}.sphere.reg" if rel == "PRED" else (sub_dir / rel if rel else None)
        if path is not None and not path.exists():
            break
        spheres[method] = nib.freesurfer.read_geometry(str(path))[0] if path else template_sphere
    if len(spheres) < len(METHODS):
        print(f"  {sub}: no {path.name}")
        continue

    line = f"  {sub}"
    for method, sphere in spheres.items():
        pred31 = atlas31[atlas_tree.query(unit(sphere))[1]]
        pred = {"DKT31": pred31[mask], "DKT25": to_dkt25(pred31)[mask]}
        for p, regions in PROTOCOLS.items():
            per_region, g = dice_scores(pred[p], manual[p], regions)
            rows[p].append(dict(subject=sub, method=method, **per_region,
                                mean_dice=float(np.nanmean(list(per_region.values()))),
                                global_dice=g))
        line += f"  {method}: {rows['DKT31'][-1]['mean_dice']:.3f}"
    print(line)

n_subj = len({r["subject"] for r in rows["DKT31"]})
print(f"\n{n_subj} subjects evaluated")
if n_subj == 0:
    raise RuntimeError("No subject evaluated.")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def values(p, method, key):
    return np.array([r[key] for r in rows[p] if r["method"] == method], float)


summary = [f"=== Dice vs MANUAL labels, Mindboggle-101 {HEMI}, {n_subj} subjects ===\n"]
for p, regions in PROTOCOLS.items():
    with open(OUTPUT_DIR / f"dice_{p}_per_subject.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "method", *regions, "mean_dice", "global_dice"])
        w.writeheader()
        w.writerows(rows[p])

    with open(OUTPUT_DIR / f"dice_{p}_per_region.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region"] + [f"{m}_{s}" for m in METHODS for s in ("mean", "std")])
        for r in regions:
            w.writerow([r] + [f"{fn(values(p, m, r)):.4f}" for m in METHODS for fn in (np.nanmean, np.nanstd)])

    # mean +- std, then ours against every other method (paired Wilcoxon)
    summary.append(f"--- {p} ({len(regions)} regions) ---")
    for key in ("mean_dice", "global_dice"):
        summary.append(f"{key}:")
        for m in METHODS:
            v = values(p, m, key)
            summary.append(f"  {m:<11} {v.mean():.4f} +/- {v.std():.4f}")
        mine = values(p, "mine", key)
        for m in METHODS:
            if m == "mine":
                continue
            other = values(p, m, key)
            diff = mine - other
            summary.append(f"  mine - {m:<11} {diff.mean():+.4f}   mine better in "
                           f"{int((diff > 0).sum())}/{len(diff)}   Wilcoxon p = {wilcoxon(mine, other).pvalue:.2e}")
    summary.append("")

summary_txt = "\n".join(summary)
(OUTPUT_DIR / "summary.txt").write_text(summary_txt)
print("\n" + summary_txt)

# mean Dice per subject, one panel per protocol
PLOT_NAMES = {"map": "No registration", "josa": "JOSA", "mine": "Proposed", "freesurfer": "FreeSurfer"}
fig, axes = plt.subplots(1, len(PROTOCOLS), figsize=(10, 4.5), sharey=True)
rng = np.random.default_rng(0)
for ax, p in zip(axes, PROTOCOLS):
    data = [values(p, m, "mean_dice") for m in METHODS]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#0b0b0b", linewidth=1.5))
    for box, m in zip(bp["boxes"], METHODS):
        box.set_facecolor(COLORS[m]); box.set_alpha(0.35); box.set_edgecolor(COLORS[m])
    for i, (m, v) in enumerate(zip(METHODS, data), 1):
        ax.scatter(rng.normal(i, 0.05, len(v)), v, s=10, color=COLORS[m], alpha=0.7, linewidths=0)
    ax.set_xticks(range(1, len(METHODS) + 1), [PLOT_NAMES[m] for m in METHODS])
    ax.set_title(f"{p} ({len(PROTOCOLS[p])} regions)")
    ax.grid(True, axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Mean Dice vs manual labels")
fig.suptitle(f"Mindboggle-101 ({HEMI}, {n_subj} subjects)")
plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUTPUT_DIR / f"dice_vs_manual_boxplot.{ext}", dpi=150)
plt.close(fig)

print(f"results in {OUTPUT_DIR}")
