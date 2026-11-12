"""
Dice of the aparc parcellation, with FreeSurfer as reference.

The fsaverage aparc (atlas.py) is moved by nearest neighbour onto the sphere of each
subject registered with our network, with JOSA and with FreeSurfer, and onto the
undeformed template (no registration: shows whether a subject is already hard
before registering). Each one is then compared with the FreeSurfer labels.

Outputs, in <run_dir>/parcel_dice:
  dice_{mine,josa,initial}_vs_fs.csv   one row per subject, Dice of every region +
                                       simple mean, size-weighted mean, global Dice
  dice_per_region_summary.csv
  who_wins.txt                         global Dice, ours vs JOSA
  outlier_subjects_{mine,diff}.csv     IQR outliers of our Dice / of ours - JOSA
  boxplots

  python -m experiments.spherical_registration.evaluation.parcel_dice --run_dir runs/ico6_train

Reads <run_dir>/test_predictions, or --pred_dir.
"""

from pathlib import Path
import argparse
import csv
import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

import brainsynth.resources

from experiments.spherical_registration.evaluation.atlas import add_fsaverage_argument, load_fsaverage_atlas
from experiments.spherical_registration import finetune_train

DATA_ROOT = Path(finetune_train.DATA_ROOT)


HEMI = "lh"

JOSA_SPHERE_NAME   = f"{HEMI}.sphere.reg.josa"
FS_SPHERE_NAME     = f"{HEMI}.sphere.reg.freesurfer"

parser = argparse.ArgumentParser(description="Parcellation Dice of the predicted and JOSA spheres vs FreeSurfer.")
parser.add_argument("--run_dir", default="runs/ico6_train",
                    help="run folder; results go in <run_dir>/parcel_dice")
parser.add_argument("--pred_dir", default=None,
                    help="folder with the predicted spheres (default: <run_dir>/test_predictions)")
add_fsaverage_argument(parser)
args = parser.parse_args()

PRED_SPHERES_DIR = Path(args.pred_dir) if args.pred_dir else Path(args.run_dir) / "test_predictions"
OUTPUT_DIR = Path(args.run_dir) / "parcel_dice"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_VERTS_EXPECTED = 245762

IQR_K = 1.5   # same rule as the whiskers of matplotlib boxplots


def normalize_to_unit_sphere(verts: np.ndarray) -> np.ndarray:
    return verts / np.linalg.norm(verts, axis=1, keepdims=True)


def transfer_labels(source_sphere, source_labels, target_sphere):
    src = normalize_to_unit_sphere(source_sphere)
    tgt = normalize_to_unit_sphere(target_sphere)
    tree = cKDTree(src)
    _, nn = tree.query(tgt)
    return source_labels[nn]


def dice_per_region(labels_a, labels_b, region_names):
    out = {}
    for lid in np.union1d(np.unique(labels_a), np.unique(labels_b)):
        if lid <= 0:
            continue
        a = labels_a == lid
        b = labels_b == lid
        denom = a.sum() + b.sum()
        if denom == 0:
            continue
        inter = np.logical_and(a, b).sum()
        name = (region_names[lid].decode()
                if lid < len(region_names) else f"label_{lid}")
        out[name] = (2.0 * inter / denom, int(b.sum()))
    return out


def simple_mean_dice(dice_dict):
    dices = [d for d, _ in dice_dict.values()]
    return float(np.nanmean(dices)) if dices else np.nan


def weighted_mean_dice(dice_dict):
    dices = np.array([d for d, _ in dice_dict.values()], float)
    sizes = np.array([s for _, s in dice_dict.values()], float)
    return float(np.average(dices, weights=sizes)) if sizes.sum() > 0 else np.nan


def global_dice(labels_a, labels_b):
    num = 0
    den = 0
    for lid in np.union1d(np.unique(labels_a), np.unique(labels_b)):
        if lid <= 0:
            continue
        a = labels_a == lid
        b = labels_b == lid
        num += 2 * np.logical_and(a, b).sum()
        den += a.sum() + b.sum()
    return float(num / den) if den > 0 else np.nan


def parse_subject_from_filename(fname: str):
    stem = fname
    suffix = f"_{HEMI}.sphere.reg"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    dataset, _, subject = stem.partition("_")
    return dataset, subject


def detect_outliers_iqr(values: np.ndarray, k: float = IQR_K):
    """Standard IQR rule (same as matplotlib boxplot whiskers).
    Returns a boolean mask: True where the value is an outlier (low OR high).
    """
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (values < lower) | (values > upper)


print(f"atlas: {args.fsaverage_dir}")
fsavg_sphere, fsavg_labels, region_names = load_fsaverage_atlas(args.fsaverage_dir, HEMI)
print(f"  {fsavg_sphere.shape[0]} vertices, {len(region_names)} regions")


# no-registration baseline: the same labels for every subject
template_verts = brainsynth.resources.Template().load_surface(HEMI, "sphere.reg")["vertices"].numpy()
template_verts = template_verts[:N_VERTS_EXPECTED]

labels_initial = transfer_labels(fsavg_sphere, fsavg_labels, template_verts)


# per-region Dice of the baseline, filled by evaluate_subject()
dice_initial_by_subject = {}


def evaluate_subject(pred_path: Path):
    """(dataset, subject, dice_mine, dice_josa, g_initial), or None if a file is missing."""
    dataset, subject = parse_subject_from_filename(pred_path.name)

    sub_dir = DATA_ROOT / dataset / subject
    josa_path   = sub_dir / "derivatives" / JOSA_SPHERE_NAME
    fs_path     = sub_dir / "derivatives" / FS_SPHERE_NAME

    if not josa_path.exists():
        print(f"  {dataset}/{subject}: no JOSA sphere")
        return None
    if not fs_path.exists():
        print(f"  {dataset}/{subject}: no FreeSurfer sphere")
        return None

    mine_sphere,   _ = nib.freesurfer.read_geometry(str(pred_path))
    josa_sphere, _ = nib.freesurfer.read_geometry(str(josa_path))
    fs_sphere,     _ = nib.freesurfer.read_geometry(str(fs_path))

    for name, s in [("mine", mine_sphere), ("JOSA", josa_sphere), ("FS", fs_sphere)]:
        if s.shape[0] != N_VERTS_EXPECTED:
            print(f"  {dataset}/{subject}: {name} sphere has {s.shape[0]} vertices")
            return None

    labels_mine   = transfer_labels(fsavg_sphere, fsavg_labels, mine_sphere)
    labels_josa = transfer_labels(fsavg_sphere, fsavg_labels, josa_sphere)
    labels_fs     = transfer_labels(fsavg_sphere, fsavg_labels, fs_sphere)

    dice_mine   = dice_per_region(labels_mine,   labels_fs, region_names)
    dice_josa = dice_per_region(labels_josa, labels_fs, region_names)

    g_mine    = global_dice(labels_mine,    labels_fs)
    g_josa  = global_dice(labels_josa,  labels_fs)
    g_initial = global_dice(labels_initial, labels_fs)

    dice_mine["__global__"]   = (g_mine, 0)
    dice_josa["__global__"] = (g_josa, 0)

    dice_initial = dice_per_region(labels_initial, labels_fs, region_names)
    dice_initial["__global__"] = (g_initial, 0)
    dice_initial_by_subject[(dataset, subject)] = dice_initial

    print(f"{dataset}/{subject}  initial {g_initial:.4f}  mine {g_mine:.4f}  josa {g_josa:.4f}")

    return dataset, subject, dice_mine, dice_josa, g_initial


pred_files = sorted(PRED_SPHERES_DIR.glob(f"*_{HEMI}.sphere.reg"))
print(f"\n{len(pred_files)} spheres in {PRED_SPHERES_DIR}")
if not pred_files:
    raise RuntimeError("No predicted spheres found. Check PRED_SPHERES_DIR and naming.")

results = []
for i, pf in enumerate(pred_files, 1):
    if i % 50 == 0 or i == 1:
        print(f"  [{i}/{len(pred_files)}] {pf.name}")
    r = evaluate_subject(pf)
    if r is not None:
        results.append(r)

print(f"\n{len(results)} / {len(pred_files)} subjects evaluated")
if not results:
    raise RuntimeError("No subjects evaluated. Check JOSA/FS paths.")

all_regions = sorted({
    r for _, _, dm, ds, _ in results
    for r in dm.keys() if r != "__global__"
})


def write_per_subject_csv(path, method):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "subject"] + all_regions
                   + ["simple_mean", "weighted_mean", "global_dice"])
        for dataset, subject, dice_mine, dice_josa, g_initial in results:
            d = {"mine": dice_mine, "josa": dice_josa,
                 "initial": dice_initial_by_subject[(dataset, subject)]}[method]
            region_vals = [d[r][0] if r in d else np.nan for r in all_regions]
            d_regions = {k: v for k, v in d.items() if k != "__global__"}
            w.writerow(
                [dataset, subject] + region_vals
                + [simple_mean_dice(d_regions),
                   weighted_mean_dice(d_regions),
                   d["__global__"][0]]
            )

for method in ("mine", "josa", "initial"):
    write_per_subject_csv(OUTPUT_DIR / f"dice_{method}_vs_fs.csv", method)


with open(OUTPUT_DIR / "dice_per_region_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["region", "mine_mean", "mine_std",
                "josa_mean", "josa_std", "mine_minus_josa"])
    for r in all_regions:
        mine = np.array([dm[r][0] for _, _, dm, _, _ in results if r in dm], float)
        josa = np.array([ds[r][0] for _, _, _, ds, _ in results if r in ds], float)
        mm, jm = np.nanmean(mine), np.nanmean(josa)
        w.writerow([r, mm, np.nanstd(mine), jm, np.nanstd(josa), mm - jm])


mine_global    = np.array([dm["__global__"][0] for _, _, dm, _, _ in results], float)
josa_global    = np.array([ds["__global__"][0] for _, _, _, ds, _ in results], float)
initial_global = np.array([g_init for _, _, _, _, g_init in results], float)

wins_mine   = int((mine_global > josa_global).sum())
wins_josa   = int((josa_global > mine_global).sum())
diff = mine_global - josa_global

summary_txt = OUTPUT_DIR / "who_wins.txt"
with open(summary_txt, "w") as f:
    f.write("=== Comparison vs FreeSurfer (ground truth) -- GLOBAL Dice ===\n\n")
    f.write(f"Subjects evaluated: {len(results)}\n\n")
    f.write(f"Global Dice (Initial, no reg.): {initial_global.mean():.4f} +/- {initial_global.std():.4f}\n")
    f.write(f"Global Dice (Mine)            : {mine_global.mean():.4f} +/- {mine_global.std():.4f}\n")
    f.write(f"Global Dice (JOSA)            : {josa_global.mean():.4f} +/- {josa_global.std():.4f}\n\n")
    f.write(f"Mean difference (Mine - JOSA): {diff.mean():+.4f}\n\n")
    f.write(f"Subjects where I win    : {wins_mine}/{len(results)}\n")
    f.write(f"Subjects where JOSA wins  : {wins_josa}/{len(results)}\n")


print("\nglobal Dice")
print(f"  Initial (no reg.): {initial_global.mean():.4f} +/- {initial_global.std():.4f}")
print(f"  Mine             : {mine_global.mean():.4f} +/- {mine_global.std():.4f}")
print(f"  JOSA             : {josa_global.mean():.4f} +/- {josa_global.std():.4f}")
print(f"  mine - JOSA      : {diff.mean():+.4f}, better on {wins_mine}/{len(results)} subjects")


outlier_mask_mine = detect_outliers_iqr(mine_global)
outlier_mask_diff = detect_outliers_iqr(diff)

def write_outliers_csv(path, mask, sort_key_idx):
    idxs = np.where(mask)[0]
    rows = []
    for idx in idxs:
        dataset, subject, dm, ds, g_init = results[idx]
        m = dm["__global__"][0]
        j = ds["__global__"][0]
        rows.append((dataset, subject, g_init, m, j, m - j))
    rows.sort(key=lambda r: r[sort_key_idx])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "subject", "initial_global_dice",
                    "mine_global_dice", "josa_global_dice", "diff"])
        w.writerows(rows)
    return rows

outliers_mine_path = OUTPUT_DIR / "outlier_subjects_mine.csv"
outliers_diff_path = OUTPUT_DIR / "outlier_subjects_diff.csv"

# sort_key_idx: 0=dataset,1=subject,2=initial,3=mine,4=josa,5=diff
rows_outliers_mine = write_outliers_csv(outliers_mine_path, outlier_mask_mine, sort_key_idx=3)
rows_outliers_diff = write_outliers_csv(outliers_diff_path, outlier_mask_diff, sort_key_idx=5)

print(f"\nOutliers on my Dice (IQR, k={IQR_K}): {outlier_mask_mine.sum()}/{len(results)} subjects")

print(f"Outliers on the mine-josa difference (IQR, k={IQR_K}): {outlier_mask_diff.sum()}/{len(results)} subjects")


worst_txt = OUTPUT_DIR / "worst_subjects.txt"
with open(worst_txt, "w") as f:
    f.write(f"=== OUTLIER subjects (IQR rule, k={IQR_K}) on MY global Dice ===\n")
    f.write(f"{'dataset':<14} {'subject':<14} {'initial':>8} {'mine':>8} {'josa':>8} {'diff':>8}\n")
    for ds_name, subj, g_init, m, j, d in rows_outliers_mine:
        f.write(f"{ds_name:<14} {subj:<14} {g_init:>8.4f} {m:>8.4f} {j:>8.4f} {d:>+8.4f}\n")

    f.write(f"\n=== OUTLIER subjects (IQR rule, k={IQR_K}) on (mine - josa) difference ===\n")
    f.write(f"{'dataset':<14} {'subject':<14} {'initial':>8} {'mine':>8} {'josa':>8} {'diff':>8}\n")
    for ds_name, subj, g_init, m, j, d in rows_outliers_diff:
        f.write(f"{ds_name:<14} {subj:<14} {g_init:>8.4f} {m:>8.4f} {j:>8.4f} {d:>+8.4f}\n")


print("\nlowest Dice:")
for ds_name, subj, g_init, m, j, d in rows_outliers_mine[:5]:
    print(f"  {ds_name}/{subj}: initial={g_init:.4f}  mine={m:.4f}  josa={j:.4f}  diff={d:+.4f}")


# one boxplot per method, outliers in red
fig, ax = plt.subplots(figsize=(4.5, 6))
ax.boxplot([initial_global], labels=["Initial (no reg.)"], showfliers=True,
           flierprops=dict(marker="o", markerfacecolor="red",
                            markersize=6, alpha=0.6, markeredgecolor="none"))
xs = np.random.normal(1, 0.04, size=len(initial_global))
ax.scatter(xs, initial_global, alpha=0.25, s=15, color="gray")
ax.set_ylabel("Global Dice vs FreeSurfer")
ax.set_title("Initial (undeformed) vs FreeSurfer (outliers in red)")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "dice_initial_vs_fs_boxplot.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.5, 6))
ax.boxplot([mine_global], labels=["Mine"], showfliers=True,
           flierprops=dict(marker="o", markerfacecolor="red",
                            markersize=6, alpha=0.6, markeredgecolor="none"))
xs = np.random.normal(1, 0.04, size=len(mine_global))
ax.scatter(xs, mine_global, alpha=0.25, s=15, color="gray")
ax.set_ylabel("Global Dice vs FreeSurfer")
ax.set_title("Mine vs FreeSurfer (outliers in red)")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "dice_mine_vs_fs_boxplot.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.5, 6))
ax.boxplot([josa_global], labels=["JOSA"], showfliers=True,
           flierprops=dict(marker="o", markerfacecolor="red",
                            markersize=6, alpha=0.6, markeredgecolor="none"))
xs = np.random.normal(1, 0.04, size=len(josa_global))
ax.scatter(xs, josa_global, alpha=0.25, s=15, color="gray")
ax.set_ylabel("Global Dice vs FreeSurfer")
ax.set_title("JOSA vs FreeSurfer (outliers in red)")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "dice_josa_vs_fs_boxplot.png", dpi=150)
plt.close(fig)

# per region, ours and JOSA side by side
fig, ax = plt.subplots(figsize=(max(14, len(all_regions) * 0.8), 6))
positions_mine = np.arange(len(all_regions)) * 2.0
positions_josa = positions_mine + 0.8
data_mine = [np.array([dm[r][0] for _, _, dm, _, _ in results if r in dm], float)
             for r in all_regions]
data_josa = [np.array([ds[r][0] for _, _, _, ds, _ in results if r in ds], float)
             for r in all_regions]
data_mine = [v[~np.isnan(v)] for v in data_mine]
data_josa = [v[~np.isnan(v)] for v in data_josa]
bp1 = ax.boxplot(data_mine, positions=positions_mine, widths=0.7,
                 patch_artist=True, showfliers=False)
bp2 = ax.boxplot(data_josa, positions=positions_josa, widths=0.7,
                 patch_artist=True, showfliers=False)
for b in bp1["boxes"]:
    b.set_facecolor("#378ADD"); b.set_alpha(0.6)
for b in bp2["boxes"]:
    b.set_facecolor("#D85A30"); b.set_alpha(0.6)
ax.set_xticks(positions_mine + 0.4)
ax.set_xticklabels(all_regions, rotation=90)
ax.set_ylabel("Dice vs FreeSurfer")
ax.set_title("Per-region Dice vs FreeSurfer (blue = Mine, orange = JOSA)")
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "dice_per_region_comparison.png", dpi=150)
plt.close(fig)

print(f"\nresults in {OUTPUT_DIR}")