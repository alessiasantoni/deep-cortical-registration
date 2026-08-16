"""
Parcellation comparison via Dice, with FreeSurfer as ground truth.

For each subject, three registered spheres (same ico=6 mesh, 245762 vertices):
  - mine (from my model)
  - Jesper's (lh.sphere.reg in the subject root)
  - FreeSurfer's (derivatives/lh.sphere.reg.freesurfer)

The SAME fsaverage atlas is transferred to each sphere (nearest-neighbor on the
unit sphere), then:
  - Dice(mine, FS)    -> how accurate I am
  - Dice(Jesper, FS)  -> how accurate Jesper is
FS is the yardstick: higher Dice = closer to the gold standard.

Three Dice summaries per subject:
  - simple mean   : unweighted mean of per-region Dice
  - weighted mean : per-region Dice weighted by region size (metric A)
  - global Dice   : micro-average over all regions (metric B) <- main metric

Output:
  - per-subject CSV (both comparisons), with all three summaries
  - per-region summary CSV
  - global comparison boxplot (mine vs Jesper)
  - per-region comparison boxplot
  - "who wins" stats based on GLOBAL Dice
"""

from pathlib import Path
import csv
import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


# ============================================================
# CONFIG  --  CHECK/FILL THESE
# ============================================================

HEMI = "lh"

# --- fsaverage atlas: BOTH the sphere AND the annotation are needed. ---
# They must be the SAME fsaverage the three spheres register to.
# Check paths with: find experiments/sphere_reg2/fsaverage_aparc/ -type f
FSAVERAGE_SPHERE = Path("experiments/sphere_reg2/fsaverage_aparc/lh.fsaverage.sphere")
FSAVERAGE_ANNOT  = Path("experiments/sphere_reg2/fsaverage_aparc/lh.aparc.annot")

# --- My predicted spheres (ico=6 run). ---
# Expected file naming: {DATASET}_{subject}_{HEMI}.sphere.reg
PRED_SPHERES_DIR = Path(
    "experiments/sphere_reg2/runs/20260609_181249_finetune_ico6/spheres_all"
)

# --- Subject data root (Jesper + FreeSurfer). ---
DATA_ROOT = Path("/projects/brainnet-data/sphere-reg")
JESPER_SPHERE_NAME = f"{HEMI}.sphere.reg"                   # in the subject root
FS_SPHERE_NAME     = f"{HEMI}.sphere.reg.freesurfer"       # in derivatives/

# --- Output ---
OUTPUT_DIR = Path("experiments/sphere_reg2/eval_parcellation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_VERTS_EXPECTED = 245762   # ico=6; all three subject spheres must match this


# ============================================================
# Helpers
# ============================================================

def normalize_to_unit_sphere(verts: np.ndarray) -> np.ndarray:
    return verts / np.linalg.norm(verts, axis=1, keepdims=True)


def transfer_labels(source_sphere: np.ndarray,
                    source_labels: np.ndarray,
                    target_sphere: np.ndarray) -> np.ndarray:
    """For each target vertex, the label of the nearest source vertex on the sphere."""
    src = normalize_to_unit_sphere(source_sphere)
    tgt = normalize_to_unit_sphere(target_sphere)
    tree = cKDTree(src)
    _, nn = tree.query(tgt)
    return source_labels[nn]


def dice_per_region(labels_a: np.ndarray, labels_b: np.ndarray,
                    region_names: list) -> dict:
    """Per-region Dice. labels_b is the ground truth (FS).

    Returns {region_name: (dice, size_gt)} where size_gt = number of
    vertices of that region in the ground truth (used as the area weight).
    Skips id <= 0 (background/unknown).
    """
    out = {}
    for lid in np.union1d(np.unique(labels_a), np.unique(labels_b)):
        if lid <= 0:
            continue
        a = labels_a == lid
        b = labels_b == lid          # ground truth (FS)
        denom = a.sum() + b.sum()
        if denom == 0:
            continue
        inter = np.logical_and(a, b).sum()
        name = (region_names[lid].decode()
                if lid < len(region_names) else f"label_{lid}")
        out[name] = (2.0 * inter / denom, int(b.sum()))
    return out


def simple_mean_dice(dice_dict: dict) -> float:
    """Unweighted mean of per-region Dice."""
    dices = [d for d, _ in dice_dict.values()]
    return float(np.nanmean(dices)) if dices else np.nan


def weighted_mean_dice(dice_dict: dict) -> float:
    """Mean of per-region Dice weighted by region size (metric A)."""
    dices = np.array([d for d, _ in dice_dict.values()], float)
    sizes = np.array([s for _, s in dice_dict.values()], float)
    return float(np.average(dices, weights=sizes)) if sizes.sum() > 0 else np.nan


def global_dice(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Global / micro-average Dice over all regions (metric B).

    Sums intersections and cardinalities across regions before dividing,
    which intrinsically weights each region by its size.
    """
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


def parse_subject_from_filename(fname: str) -> tuple[str, str]:
    """'ABIDE_sub-0001_lh.sphere.reg' -> ('ABIDE', 'sub-0001')."""
    stem = fname
    suffix = f"_{HEMI}.sphere.reg"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    dataset, _, subject = stem.partition("_")   # split on the FIRST underscore
    return dataset, subject


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

# CRITICAL CHECK: sphere and annot from the same fsaverage
assert fsavg_sphere.shape[0] == len(fsavg_labels), (
    f"fsaverage MISMATCH: sphere has {fsavg_sphere.shape[0]} vertices but "
    f"annot has {len(fsavg_labels)}. They are not the same fsaverage!"
)


# ============================================================
# Per-subject evaluation
# ============================================================

def evaluate_subject(pred_path: Path):
    """Returns (dataset, subject, dice_mine, dice_jesper) or None if files missing.

    dice_mine / dice_jesper are dicts {region: (dice, size_gt)}.
    Also stores the global Dice on the dict under key '__global__'.
    """
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

    # Load the three spheres (same ico=6 mesh)
    mine_sphere,   _ = nib.freesurfer.read_geometry(str(pred_path))
    jesper_sphere, _ = nib.freesurfer.read_geometry(str(jesper_path))
    fs_sphere,     _ = nib.freesurfer.read_geometry(str(fs_path))

    # Sanity: same vertex count (vertex-wise comparison valid)
    for name, s in [("mine", mine_sphere), ("Jesper", jesper_sphere), ("FS", fs_sphere)]:
        if s.shape[0] != N_VERTS_EXPECTED:
            print(f"  [{dataset}/{subject}] {name} sphere has {s.shape[0]} "
                  f"vertices (expected {N_VERTS_EXPECTED}), skip")
            return None

    # Transfer the SAME atlas to each sphere
    labels_mine   = transfer_labels(fsavg_sphere, fsavg_labels, mine_sphere)
    labels_jesper = transfer_labels(fsavg_sphere, fsavg_labels, jesper_sphere)
    labels_fs     = transfer_labels(fsavg_sphere, fsavg_labels, fs_sphere)

    # Per-region Dice with FS as ground truth
    dice_mine   = dice_per_region(labels_mine,   labels_fs, region_names)
    dice_jesper = dice_per_region(labels_jesper, labels_fs, region_names)

    # Global Dice (metric B) — the main summary metric
    g_mine   = global_dice(labels_mine,   labels_fs)
    g_jesper = global_dice(labels_jesper, labels_fs)
    dice_mine["__global__"]   = (g_mine, 0)
    dice_jesper["__global__"] = (g_jesper, 0)

    print(
        f"[{dataset}/{subject}]  "
        f"Mine: {g_mine:.4f} | Jesper: {g_jesper:.4f} | "
        f"Diff: {g_mine - g_jesper:+.4f}  (global Dice)"
    )

    return dataset, subject, dice_mine, dice_jesper


# ============================================================
# Loop over subjects
# ============================================================

pred_files = sorted(PRED_SPHERES_DIR.glob(f"*_{HEMI}.sphere.reg"))
print(f"\nFound {len(pred_files)} predicted spheres in {PRED_SPHERES_DIR}")
if not pred_files:
    raise RuntimeError("No predicted spheres found. Check PRED_SPHERES_DIR and naming.")

results = []   # (dataset, subject, dice_mine, dice_jesper)
for i, pf in enumerate(pred_files, 1):
    if i % 50 == 0 or i == 1:
        print(f"  [{i}/{len(pred_files)}] {pf.name}")
    r = evaluate_subject(pf)
    if r is not None:
        results.append(r)

print(f"\nEvaluated: {len(results)} / {len(pred_files)}")
if not results:
    raise RuntimeError("No subjects evaluated. Check Jesper/FS paths.")


# Region names (exclude the internal '__global__' key)
all_regions = sorted({
    r for *_, dm, ds in results
    for r in dm.keys() if r != "__global__"
})


# ============================================================
# Output 1: per-subject CSV (both comparisons), all three summaries
# ============================================================

def write_per_subject_csv(path, idx):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "subject"] + all_regions
                   + ["simple_mean", "weighted_mean", "global_dice"])
        for dataset, subject, dice_mine, dice_jesper in results:
            d = (dice_mine, dice_jesper)[idx]
            region_vals = [d[r][0] if r in d else np.nan for r in all_regions]
            # exclude '__global__' from the per-region summaries
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
        mine = np.array([dm[r][0] for *_, dm, _ in results if r in dm], float)
        jesp = np.array([ds[r][0] for *_, _, ds in results if r in ds], float)
        mm, jm = np.nanmean(mine), np.nanmean(jesp)
        w.writerow([r, mm, np.nanstd(mine), jm, np.nanstd(jesp), mm - jm])
print(f"Per-region summary -> {OUTPUT_DIR}/dice_per_region_summary.csv")


# ============================================================
# Output 3: "who wins" -- based on GLOBAL Dice (metric B)
# ============================================================

mine_global   = np.array([dm["__global__"][0] for *_, dm, _ in results], float)
jesper_global = np.array([ds["__global__"][0] for *_, _, ds in results], float)

wins_mine   = int((mine_global > jesper_global).sum())
wins_jesper = int((jesper_global > mine_global).sum())
diff = mine_global - jesper_global

summary_txt = OUTPUT_DIR / "who_wins.txt"
with open(summary_txt, "w") as f:
    f.write("=== Comparison vs FreeSurfer (ground truth) -- GLOBAL Dice ===\n\n")
    f.write(f"Subjects evaluated: {len(results)}\n\n")
    f.write(f"Global Dice (Mine)   : {mine_global.mean():.4f} +/- {mine_global.std():.4f}\n")
    f.write(f"Global Dice (Jesper) : {jesper_global.mean():.4f} +/- {jesper_global.std():.4f}\n\n")
    f.write(f"Mean difference (Mine - Jesper): {diff.mean():+.4f}\n\n")
    f.write(f"Subjects where I win    : {wins_mine}/{len(results)}\n")
    f.write(f"Subjects where Jesper wins: {wins_jesper}/{len(results)}\n")
print(f"Who wins -> {summary_txt}")

print("\n=== RESULT (global Dice) ===")
print(f"  Mine   : {mine_global.mean():.4f} +/- {mine_global.std():.4f}")
print(f"  Jesper : {jesper_global.mean():.4f} +/- {jesper_global.std():.4f}")
print(f"  Diff   : {diff.mean():+.4f}  (I win {wins_mine}/{len(results)} subjects)")

# ============================================================
# Output 3b: best / worst subjects
# ============================================================

# Lista per-soggetto: (dataset, subject, mine_global, jesper_global, diff)
per_subject = [
    (dataset, subject,
     dm["__global__"][0],
     ds["__global__"][0],
     dm["__global__"][0] - ds["__global__"][0])
    for dataset, subject, dm, ds in results
]

N_SHOW = 10   # quanti soggetti mostrare in ogni classifica

worst_txt = OUTPUT_DIR / "best_worst_subjects.txt"
with open(worst_txt, "w") as f:

    # --- 1. Worst by MY global Dice (where I struggle in absolute terms) ---
    by_mine = sorted(per_subject, key=lambda x: x[2])   # ascending: worst first
    f.write("=== Worst subjects by MY global Dice (lowest first) ===\n")
    f.write(f"{'dataset':<10} {'subject':<14} {'mine':>8} {'jesper':>8} {'diff':>8}\n")
    for ds_name, subj, m, j, d in by_mine[:N_SHOW]:
        f.write(f"{ds_name:<10} {subj:<14} {m:>8.4f} {j:>8.4f} {d:>+8.4f}\n")

    f.write("\n=== Best subjects by MY global Dice (highest first) ===\n")
    f.write(f"{'dataset':<10} {'subject':<14} {'mine':>8} {'jesper':>8} {'diff':>8}\n")
    for ds_name, subj, m, j, d in by_mine[-N_SHOW:][::-1]:
        f.write(f"{ds_name:<10} {subj:<14} {m:>8.4f} {j:>8.4f} {d:>+8.4f}\n")

    # --- 2. Worst by DIFFERENCE (where Jesper beats me the most) ---
    by_diff = sorted(per_subject, key=lambda x: x[4])   # ascending: most negative first
    f.write("\n=== Subjects where Jesper beats me the most (diff = mine - jesper) ===\n")
    f.write(f"{'dataset':<10} {'subject':<14} {'mine':>8} {'jesper':>8} {'diff':>8}\n")
    for ds_name, subj, m, j, d in by_diff[:N_SHOW]:
        f.write(f"{ds_name:<10} {subj:<14} {m:>8.4f} {j:>8.4f} {d:>+8.4f}\n")

    f.write("\n=== Subjects where I beat Jesper the most ===\n")
    f.write(f"{'dataset':<10} {'subject':<14} {'mine':>8} {'jesper':>8} {'diff':>8}\n")
    for ds_name, subj, m, j, d in by_diff[-N_SHOW:][::-1]:
        f.write(f"{ds_name:<10} {subj:<14} {m:>8.4f} {j:>8.4f} {d:>+8.4f}\n")

print(f"Best/worst subjects -> {worst_txt}")

# Print the worst few to console for a quick look
print(f"\n=== Worst {min(5, len(by_mine))} subjects (my global Dice) ===")
for ds_name, subj, m, j, d in by_mine[:5]:
    print(f"  {ds_name}/{subj}: mine={m:.4f}  jesper={j:.4f}  diff={d:+.4f}")

# ============================================================
# Output 4: plots
# ============================================================

# (a) Global boxplot: global Dice per subject, mine vs Jesper
fig, ax = plt.subplots(figsize=(7, 6))
ax.boxplot([mine_global, jesper_global], labels=["Mine", "Jesper"], showfliers=False)
for j, vals in enumerate([mine_global, jesper_global], start=1):
    xs = np.random.normal(j, 0.04, size=len(vals))
    ax.scatter(xs, vals, alpha=0.4, s=20, color="gray")
ax.set_ylabel("Global Dice vs FreeSurfer")
ax.set_title("Accuracy vs FreeSurfer: Mine vs Jesper")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "dice_global_comparison.png", dpi=150)
plt.close(fig)

# (b) Per-region boxplot, side by side (mine vs Jesper)
fig, ax = plt.subplots(figsize=(max(14, len(all_regions) * 0.8), 6))
positions_mine = np.arange(len(all_regions)) * 2.0
positions_jesp = positions_mine + 0.8
data_mine = [np.array([dm[r][0] for *_, dm, _ in results if r in dm], float)
             for r in all_regions]
data_jesp = [np.array([ds[r][0] for *_, _, ds in results if r in ds], float)
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