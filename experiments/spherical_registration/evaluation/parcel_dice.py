"""
Parcellation comparison via Dice, with FreeSurfer as ground truth.

For each subject, THREE registered spheres + ONE fixed baseline
(same ico=6 mesh, 245762 vertices):
  - mine (from my model)
  - Jesper's (lh.sphere.reg in the subject root)
  - FreeSurfer's (derivatives/lh.sphere.reg.freesurfer)
  - initial (undeformed template sphere, FIXED, same for every subject --
    "zero registration" baseline, used to diagnose whether outliers already
    start shifted before any registration is applied)

The SAME fsaverage atlas is transferred to each sphere (nearest-neighbor on the
unit sphere), then:
  - Dice(mine, FS)     -> how accurate I am
  - Dice(Jesper, FS)   -> how accurate Jesper is
  - Dice(initial, FS)  -> how accurate "no registration at all" already is
FS is the yardstick: higher Dice = closer to the gold standard.

Output:
  - per-subject CSV (mine/jesper), with all three summaries
  - per-region summary CSV
  - THREE separate boxplots: Initial vs FS, Mine vs FS, Jesper vs FS
  - per-region comparison boxplot
  - "who wins" stats based on GLOBAL Dice
  - outlier subjects (IQR rule) = the "worst subjects" list, ready for VTK
    visualization, now also reporting the initial Dice for diagnosis
"""

from pathlib import Path
import csv
import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

import brainsynth.resources


# ============================================================
# CONFIG  --  CHECK/FILL THESE
# ============================================================

HEMI = "lh"

FSAVERAGE_SPHERE = Path("experiments/sphere_reg2/fsaverage_aparc/lh.fsaverage.sphere")
FSAVERAGE_ANNOT  = Path("experiments/sphere_reg2/fsaverage_aparc/lh.aparc.annot")

PRED_SPHERES_DIR = Path(
    "/home/s252125/my_project/experiments/sphere_reg2/runs/ico6_full_test"
)

DATA_ROOT = Path("/projects/brainnet-data/sphere-reg")
JESPER_SPHERE_NAME = f"{HEMI}.sphere.reg"
FS_SPHERE_NAME     = f"{HEMI}.sphere.reg.freesurfer"

OUTPUT_DIR = Path(
    "/home/s252125/my_project/experiments/sphere_reg2/runs/parcel_dice"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_VERTS_EXPECTED = 245762

IQR_K = 1.5   # regola standard dei boxplot matplotlib


# ============================================================
# Helpers
# ============================================================

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
    """Regola IQR standard (la stessa dei baffi nei boxplot matplotlib).
    Ritorna una maschera booleana: True dove il valore e' outlier (basso O alto).
    """
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (values < lower) | (values > upper)


# ============================================================
# Load the fsaverage atlas (once)
# ============================================================

print("Loading fsaverage atlas")
print(f"  sphere: {FSAVERAGE_SPHERE}")
print(f"  annot:  {FSAVERAGE_ANNOT}")

fsavg_sphere, _ = nib.freesurfer.read_geometry(str(FSAVERAGE_SPHERE))
fsavg_labels, ctab, region_names = nib.freesurfer.read_annot(str(FSAVERAGE_ANNOT))

print(f"  -> sphere: {fsavg_sphere.shape[0]} vertices, "
      f"annot: {len(fsavg_labels)} vertices, {len(region_names)} regions")

assert fsavg_sphere.shape[0] == len(fsavg_labels), (
    f"fsaverage MISMATCH: sphere has {fsavg_sphere.shape[0]} vertices but "
    f"annot has {len(fsavg_labels)}. They are not the same fsaverage!"
)


# ============================================================
# Labels sulla sfera INIZIALE (fissa, non deformata, baseline "zero registrazione")
# ============================================================

print("\nComputing baseline labels on the UNDEFORMED initial sphere (same for all subjects)")
template_resources = brainsynth.resources.Template()
template_verts_full = template_resources.load_surface(HEMI, "sphere.reg")["vertices"].numpy()
template_verts = template_verts_full[:N_VERTS_EXPECTED]

labels_initial = transfer_labels(fsavg_sphere, fsavg_labels, template_verts)
print(f"  -> baseline labels computed on {template_verts.shape[0]} vertices "
      f"(fixed, no per-subject registration)")


# ============================================================
# Per-subject evaluation
# ============================================================

def evaluate_subject(pred_path: Path):
    """Returns (dataset, subject, dice_mine, dice_jesper, g_initial) or None."""
    dataset, subject = parse_subject_from_filename(pred_path.name)

    sub_dir = DATA_ROOT / dataset / subject
    jesper_path = sub_dir / JESPER_SPHERE_NAME
    fs_path     = sub_dir / "derivatives" / FS_SPHERE_NAME

    if not jesper_path.exists():
        print(f"  [{dataset}/{subject}] Jesper sphere missing, skip")
        return None
    if not fs_path.exists():
        print(f"  [{dataset}/{subject}] FreeSurfer sphere missing, skip")
        return None

    mine_sphere,   _ = nib.freesurfer.read_geometry(str(pred_path))
    jesper_sphere, _ = nib.freesurfer.read_geometry(str(jesper_path))
    fs_sphere,     _ = nib.freesurfer.read_geometry(str(fs_path))

    for name, s in [("mine", mine_sphere), ("Jesper", jesper_sphere), ("FS", fs_sphere)]:
        if s.shape[0] != N_VERTS_EXPECTED:
            print(f"  [{dataset}/{subject}] {name} sphere has {s.shape[0]} "
                  f"vertices (expected {N_VERTS_EXPECTED}), skip")
            return None

    labels_mine   = transfer_labels(fsavg_sphere, fsavg_labels, mine_sphere)
    labels_jesper = transfer_labels(fsavg_sphere, fsavg_labels, jesper_sphere)
    labels_fs     = transfer_labels(fsavg_sphere, fsavg_labels, fs_sphere)

    dice_mine   = dice_per_region(labels_mine,   labels_fs, region_names)
    dice_jesper = dice_per_region(labels_jesper, labels_fs, region_names)

    g_mine    = global_dice(labels_mine,    labels_fs)
    g_jesper  = global_dice(labels_jesper,  labels_fs)
    g_initial = global_dice(labels_initial, labels_fs)   # baseline, FS di QUESTO soggetto

    dice_mine["__global__"]   = (g_mine, 0)
    dice_jesper["__global__"] = (g_jesper, 0)

    print(
        f"[{dataset}/{subject}]  "
        f"Initial: {g_initial:.4f} | Mine: {g_mine:.4f} | Jesper: {g_jesper:.4f} | "
        f"Diff mine-jesper: {g_mine - g_jesper:+.4f}"
    )

    return dataset, subject, dice_mine, dice_jesper, g_initial


# ============================================================
# Loop over subjects
# ============================================================

pred_files = sorted(PRED_SPHERES_DIR.glob(f"*_{HEMI}.sphere.reg"))
print(f"\nFound {len(pred_files)} predicted spheres in {PRED_SPHERES_DIR}")
if not pred_files:
    raise RuntimeError("No predicted spheres found. Check PRED_SPHERES_DIR and naming.")

results = []   # list of (dataset, subject, dice_mine, dice_jesper, g_initial)
for i, pf in enumerate(pred_files, 1):
    if i % 50 == 0 or i == 1:
        print(f"  [{i}/{len(pred_files)}] {pf.name}")
    r = evaluate_subject(pf)
    if r is not None:
        results.append(r)

print(f"\nEvaluated: {len(results)} / {len(pred_files)}")
if not results:
    raise RuntimeError("No subjects evaluated. Check Jesper/FS paths.")

all_regions = sorted({
    r for _, _, dm, ds, _ in results
    for r in dm.keys() if r != "__global__"
})


# ============================================================
# Output 1: per-subject CSV (mine / jesper, con tutte e 3 le sintesi)
# ============================================================

def write_per_subject_csv(path, idx):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "subject"] + all_regions
                   + ["simple_mean", "weighted_mean", "global_dice"])
        for dataset, subject, dice_mine, dice_jesper, g_initial in results:
            d = (dice_mine, dice_jesper)[idx]
            region_vals = [d[r][0] if r in d else np.nan for r in all_regions]
            d_regions = {k: v for k, v in d.items() if k != "__global__"}
            w.writerow(
                [dataset, subject] + region_vals
                + [simple_mean_dice(d_regions),
                   weighted_mean_dice(d_regions),
                   d["__global__"][0]]
            )

write_per_subject_csv(OUTPUT_DIR / "dice_mine_vs_fs.csv", 0)
write_per_subject_csv(OUTPUT_DIR / "dice_jesper_vs_fs.csv", 1)
print(f"Per-subject CSV -> {OUTPUT_DIR}/dice_mine_vs_fs.csv, dice_jesper_vs_fs.csv")


# ============================================================
# Output 2: per-region summary
# ============================================================

with open(OUTPUT_DIR / "dice_per_region_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["region", "mine_mean", "mine_std",
                "jesper_mean", "jesper_std", "mine_minus_jesper"])
    for r in all_regions:
        mine = np.array([dm[r][0] for _, _, dm, _, _ in results if r in dm], float)
        jesp = np.array([ds[r][0] for _, _, _, ds, _ in results if r in ds], float)
        mm, jm = np.nanmean(mine), np.nanmean(jesp)
        w.writerow([r, mm, np.nanstd(mine), jm, np.nanstd(jesp), mm - jm])
print(f"Per-region summary -> {OUTPUT_DIR}/dice_per_region_summary.csv")


# ============================================================
# Output 3: "who wins" -- based on GLOBAL Dice (metric B)
# ============================================================

mine_global    = np.array([dm["__global__"][0] for _, _, dm, _, _ in results], float)
jesper_global  = np.array([ds["__global__"][0] for _, _, _, ds, _ in results], float)
initial_global = np.array([g_init for _, _, _, _, g_init in results], float)

wins_mine   = int((mine_global > jesper_global).sum())
wins_jesper = int((jesper_global > mine_global).sum())
diff = mine_global - jesper_global

summary_txt = OUTPUT_DIR / "who_wins.txt"
with open(summary_txt, "w") as f:
    f.write("=== Comparison vs FreeSurfer (ground truth) -- GLOBAL Dice ===\n\n")
    f.write(f"Subjects evaluated: {len(results)}\n\n")
    f.write(f"Global Dice (Initial, no reg.): {initial_global.mean():.4f} +/- {initial_global.std():.4f}\n")
    f.write(f"Global Dice (Mine)            : {mine_global.mean():.4f} +/- {mine_global.std():.4f}\n")
    f.write(f"Global Dice (Jesper)          : {jesper_global.mean():.4f} +/- {jesper_global.std():.4f}\n\n")
    f.write(f"Mean difference (Mine - Jesper): {diff.mean():+.4f}\n\n")
    f.write(f"Subjects where I win    : {wins_mine}/{len(results)}\n")
    f.write(f"Subjects where Jesper wins: {wins_jesper}/{len(results)}\n")
print(f"Who wins -> {summary_txt}")

print("\n=== RESULT (global Dice) ===")
print(f"  Initial (no reg.): {initial_global.mean():.4f} +/- {initial_global.std():.4f}")
print(f"  Mine             : {mine_global.mean():.4f} +/- {mine_global.std():.4f}")
print(f"  Jesper           : {jesper_global.mean():.4f} +/- {jesper_global.std():.4f}")
print(f"  Diff             : {diff.mean():+.4f}  (I win {wins_mine}/{len(results)} subjects)")


# ============================================================
# Output 3b: OUTLIER subjects (IQR rule) -- questa E' la lista "worst"
#            ora include anche il Dice INIZIALE per diagnosi
# ============================================================

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
                    "mine_global_dice", "jesper_global_dice", "diff"])
        w.writerows(rows)
    return rows

outliers_mine_path = OUTPUT_DIR / "outlier_subjects_mine.csv"
outliers_diff_path = OUTPUT_DIR / "outlier_subjects_diff.csv"

# sort_key_idx: 0=dataset,1=subject,2=initial,3=mine,4=jesper,5=diff
rows_outliers_mine = write_outliers_csv(outliers_mine_path, outlier_mask_mine, sort_key_idx=3)
rows_outliers_diff = write_outliers_csv(outliers_diff_path, outlier_mask_diff, sort_key_idx=5)

print(f"\nOutlier sul mio Dice (IQR, k={IQR_K}): {outlier_mask_mine.sum()}/{len(results)} soggetti")
print(f"  -> {outliers_mine_path}")
print(f"Outlier sulla differenza mine-jesper (IQR, k={IQR_K}): {outlier_mask_diff.sum()}/{len(results)} soggetti")
print(f"  -> {outliers_diff_path}")

worst_txt = OUTPUT_DIR / "worst_subjects.txt"
with open(worst_txt, "w") as f:
    f.write(f"=== OUTLIER subjects (IQR rule, k={IQR_K}) on MY global Dice ===\n")
    f.write(f"{'dataset':<14} {'subject':<14} {'initial':>8} {'mine':>8} {'jesper':>8} {'diff':>8}\n")
    for ds_name, subj, g_init, m, j, d in rows_outliers_mine:
        f.write(f"{ds_name:<14} {subj:<14} {g_init:>8.4f} {m:>8.4f} {j:>8.4f} {d:>+8.4f}\n")

    f.write(f"\n=== OUTLIER subjects (IQR rule, k={IQR_K}) on (mine - jesper) difference ===\n")
    f.write(f"{'dataset':<14} {'subject':<14} {'initial':>8} {'mine':>8} {'jesper':>8} {'diff':>8}\n")
    for ds_name, subj, g_init, m, j, d in rows_outliers_diff:
        f.write(f"{ds_name:<14} {subj:<14} {g_init:>8.4f} {m:>8.4f} {j:>8.4f} {d:>+8.4f}\n")

print(f"Lista worst/outlier -> {worst_txt}")

print(f"\n=== Primi outlier (mio Dice, i piu' bassi) ===")
for ds_name, subj, g_init, m, j, d in rows_outliers_mine[:5]:
    print(f"  {ds_name}/{subj}: initial={g_init:.4f}  mine={m:.4f}  jesper={j:.4f}  diff={d:+.4f}")


# ============================================================
# Output 4: plots
# ============================================================

# (a0) Boxplot separato: INITIAL (no registration) vs FS
fig, ax = plt.subplots(figsize=(4.5, 6))
ax.boxplot([initial_global], labels=["Initial (no reg.)"], showfliers=True,
           flierprops=dict(marker="o", markerfacecolor="red",
                            markersize=6, alpha=0.6, markeredgecolor="none"))
xs = np.random.normal(1, 0.04, size=len(initial_global))
ax.scatter(xs, initial_global, alpha=0.25, s=15, color="gray")
ax.set_ylabel("Global Dice vs FreeSurfer")
ax.set_title("Initial (undeformed) vs FreeSurfer (outlier in rosso)")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "dice_initial_vs_fs_boxplot.png", dpi=150)
plt.close(fig)

# (a1) Boxplot separato: MINE vs FS
fig, ax = plt.subplots(figsize=(4.5, 6))
ax.boxplot([mine_global], labels=["Mine"], showfliers=True,
           flierprops=dict(marker="o", markerfacecolor="red",
                            markersize=6, alpha=0.6, markeredgecolor="none"))
xs = np.random.normal(1, 0.04, size=len(mine_global))
ax.scatter(xs, mine_global, alpha=0.25, s=15, color="gray")
ax.set_ylabel("Global Dice vs FreeSurfer")
ax.set_title("Mine vs FreeSurfer (outlier in rosso)")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "dice_mine_vs_fs_boxplot.png", dpi=150)
plt.close(fig)

# (a2) Boxplot separato: JESPER vs FS
fig, ax = plt.subplots(figsize=(4.5, 6))
ax.boxplot([jesper_global], labels=["Jesper"], showfliers=True,
           flierprops=dict(marker="o", markerfacecolor="red",
                            markersize=6, alpha=0.6, markeredgecolor="none"))
xs = np.random.normal(1, 0.04, size=len(jesper_global))
ax.scatter(xs, jesper_global, alpha=0.25, s=15, color="gray")
ax.set_ylabel("Global Dice vs FreeSurfer")
ax.set_title("Jesper vs FreeSurfer (outlier in rosso)")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "dice_jesper_vs_fs_boxplot.png", dpi=150)
plt.close(fig)

# (b) Per-region boxplot, side by side (invariato)
fig, ax = plt.subplots(figsize=(max(14, len(all_regions) * 0.8), 6))
positions_mine = np.arange(len(all_regions)) * 2.0
positions_jesp = positions_mine + 0.8
data_mine = [np.array([dm[r][0] for _, _, dm, _, _ in results if r in dm], float)
             for r in all_regions]
data_jesp = [np.array([ds[r][0] for _, _, _, ds, _ in results if r in ds], float)
             for r in all_regions]
data_mine = [v[~np.isnan(v)] for v in data_mine]
data_jesp = [v[~np.isnan(v)] for v in data_jesp]
bp1 = ax.boxplot(data_mine, positions=positions_mine, widths=0.7,
                 patch_artist=True, showfliers=False)
bp2 = ax.boxplot(data_jesp, positions=positions_jesp, widths=0.7,
                 patch_artist=True, showfliers=False)
for b in bp1["boxes"]:
    b.set_facecolor("#378ADD"); b.set_alpha(0.6)
for b in bp2["boxes"]:
    b.set_facecolor("#D85A30"); b.set_alpha(0.6)
ax.set_xticks(positions_mine + 0.4)
ax.set_xticklabels(all_regions, rotation=90)
ax.set_ylabel("Dice vs FreeSurfer")
ax.set_title("Per-region Dice vs FreeSurfer (blue = Mine, orange = Jesper)")
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "dice_per_region_comparison.png", dpi=150)
plt.close(fig)

print(f"\nPlots -> {OUTPUT_DIR}")