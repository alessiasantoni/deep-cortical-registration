"""
Confronto di parcellazione via Dice, con FreeSurfer come ground truth.

Per ogni soggetto, tre sfere registrate (stessa mesh ico=6, 245762 vertici):
  - la mia (dal mio modello)
  - quella del supervisor (lh.sphere.reg nella root)
  - quella di FreeSurfer (derivatives/lh.sphere.reg.freesurfer)

Si trasferisce lo STESSO atlante fsaverage a ciascuna sfera (nearest-neighbor
sulla sfera unitaria), poi si calcola:
  - Dice(mia, FS)         -> quanto sono accurata io
  - Dice(supervisor, FS)  -> quanto e' accurato il supervisor
FS fa da metro: chi ha il Dice piu' alto e' piu' vicino al gold standard.

Output:
  - CSV con Dice per regione, per soggetto, per entrambi i confronti
  - CSV riassuntivo (media per regione)
  - boxplot di confronto (mia vs supervisor)
  - statistica "chi vince" (quanti soggetti, differenza media)
"""

from pathlib import Path
import csv
import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


# ============================================================
# CONFIG  --  RIEMPI/VERIFICA QUESTE RIGHE
# ============================================================

HEMI = "lh"

# --- Atlante fsaverage: servono SIA la sfera SIA le annotation. ---
# Devono essere lo STESSO fsaverage verso cui le tue 3 sfere registrano.
# Verifica i path con: find experiments/sphere_reg2/fsaverage_aparc/ -type f
FSAVERAGE_SPHERE = Path("experiments/sphere_reg2/fsaverage_aparc/lh.fsaverage.sphere")
FSAVERAGE_ANNOT  = Path("experiments/sphere_reg2/fsaverage_aparc/lh.aparc.annot")




# --- Le mie sfere predette (run ico=6). ---
# Naming atteso dei file: {DATASET}_{subject}_{HEMI}.sphere.reg
PRED_SPHERES_DIR = Path(
    "experiments/sphere_reg2/runs/20260609_181249_finetune_ico6/spheres_all"
)

# --- Root dei dati dei soggetti (supervisor + FreeSurfer). ---
DATA_ROOT = Path("/projects/brainnet-data/sphere-reg")
SUPERVISOR_SPHERE_NAME = f"{HEMI}.sphere.reg"                      # nella root del soggetto
FS_SPHERE_NAME         = f"{HEMI}.sphere.reg.freesurfer"          # in derivatives/

# --- Output ---
OUTPUT_DIR = Path("experiments/sphere_reg2/eval_parcellation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_VERTS_EXPECTED = 245762   # ico=6; le 3 sfere soggetto devono avere questo


# ============================================================
# Helpers
# ============================================================

def normalize_to_unit_sphere(verts: np.ndarray) -> np.ndarray:
    return verts / np.linalg.norm(verts, axis=1, keepdims=True)


def transfer_labels(source_sphere: np.ndarray,
                    source_labels: np.ndarray,
                    target_sphere: np.ndarray) -> np.ndarray:
    """Per ogni vertice target, la label del vertice source piu' vicino sulla sfera."""
    src = normalize_to_unit_sphere(source_sphere)
    tgt = normalize_to_unit_sphere(target_sphere)
    tree = cKDTree(src)
    _, nn = tree.query(tgt)
    return source_labels[nn]


def dice_per_region(labels_a: np.ndarray, labels_b: np.ndarray,
                    region_names: list) -> dict:
    """Dice per ogni regione (salta id <= 0 = background/unknown)."""
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
        out[name] = 2.0 * inter / denom
    return out


def parse_subject_from_filename(fname: str) -> tuple[str, str]:
    """'ABIDE_sub-0001_lh.sphere.reg' -> ('ABIDE', 'sub-0001')."""
    stem = fname
    suffix = f"_{HEMI}.sphere.reg"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    dataset, _, subject = stem.partition("_")   # split sul PRIMO underscore
    return dataset, subject


# ============================================================
# Carica l'atlante fsaverage (una sola volta)
# ============================================================

print(f"Loading fsaverage atlas")
print(f"  sphere: {FSAVERAGE_SPHERE}")
print(f"  annot:  {FSAVERAGE_ANNOT}")

fsavg_sphere, _ = nib.freesurfer.read_geometry(str(FSAVERAGE_SPHERE))
fsavg_labels, ctab, region_names = nib.freesurfer.read_annot(str(FSAVERAGE_ANNOT))

print(f"  -> sphere: {fsavg_sphere.shape[0]} vertici, "
      f"annot: {len(fsavg_labels)} vertici, {len(region_names)} regioni")

# CHECK CRITICO: sphere e annot dello stesso fsaverage
assert fsavg_sphere.shape[0] == len(fsavg_labels), (
    f"MISMATCH fsaverage: la sfera ha {fsavg_sphere.shape[0]} vertici ma "
    f"l'annot ne ha {len(fsavg_labels)}. Non sono lo stesso fsaverage!"
)


# ============================================================
# Valutazione per soggetto
# ============================================================

def evaluate_subject(pred_path: Path):
    """Ritorna (dice_mia_vs_fs, dice_sup_vs_fs) o None se mancano file."""
    dataset, subject = parse_subject_from_filename(pred_path.name)

    sub_dir = DATA_ROOT / dataset / subject
    sup_path = sub_dir / SUPERVISOR_SPHERE_NAME
    fs_path  = sub_dir / "derivatives" / FS_SPHERE_NAME

    if not sup_path.exists():
        print(f"  [{dataset}/{subject}] manca sfera supervisor, skip")
        return None
    if not fs_path.exists():
        print(f"  [{dataset}/{subject}] manca sfera freesurfer, skip")
        return None

    # Carica le 3 sfere (stessa mesh ico=6)
    mia_sphere, _ = nib.freesurfer.read_geometry(str(pred_path))
    sup_sphere, _ = nib.freesurfer.read_geometry(str(sup_path))
    fs_sphere,  _ = nib.freesurfer.read_geometry(str(fs_path))

    # Sanity: stesso numero di vertici (confronto vertice-per-vertice valido)
    for name, s in [("mia", mia_sphere), ("sup", sup_sphere), ("fs", fs_sphere)]:
        if s.shape[0] != N_VERTS_EXPECTED:
            print(f"  [{dataset}/{subject}] sfera {name} ha {s.shape[0]} "
                  f"vertici (atteso {N_VERTS_EXPECTED}), skip")
            return None

    # Trasferisci lo STESSO atlante a ciascuna sfera
    labels_mia = transfer_labels(fsavg_sphere, fsavg_labels, mia_sphere)
    labels_sup = transfer_labels(fsavg_sphere, fsavg_labels, sup_sphere)
    labels_fs  = transfer_labels(fsavg_sphere, fsavg_labels, fs_sphere)

    # Dice con FS come ground truth
    dice_mia = dice_per_region(labels_mia, labels_fs, region_names)
    dice_sup = dice_per_region(labels_sup, labels_fs, region_names)
    

    mean_dice_mia = np.nanmean(list(dice_mia.values()))
    mean_dice_sup = np.nanmean(list(dice_sup.values()))

    print(
        f"[{dataset}/{subject}] "
        f"Mia: {mean_dice_mia:.4f} | "
        f"Jesper: {mean_dice_sup:.4f} | "
        f"Diff: {mean_dice_mia - mean_dice_sup:+.4f}"
        )

    return dataset, subject, dice_mia, dice_sup


# ============================================================
# Loop sui soggetti
# ============================================================

pred_files = sorted(PRED_SPHERES_DIR.glob(f"*_{HEMI}.sphere.reg"))
print(f"\nTrovate {len(pred_files)} sfere predette in {PRED_SPHERES_DIR}")
if not pred_files:
    raise RuntimeError("Nessuna sfera predetta trovata. Controlla PRED_SPHERES_DIR e il naming.")

results = []   # (dataset, subject, dice_mia, dice_sup)
for i, pf in enumerate(pred_files, 1):
    if i % 50 == 0 or i == 1:
        print(f"  [{i}/{len(pred_files)}] {pf.name}")
    r = evaluate_subject(pf)
    if r is not None:
        results.append(r)

print(f"\nValutati: {len(results)} / {len(pred_files)}")
if not results:
    raise RuntimeError("Nessun soggetto valutato. Controlla i path supervisor/FS.")


# ============================================================
# Output 1: CSV per-soggetto (entrambi i confronti)
# ============================================================

all_regions = sorted({r for *_, dm, ds in results for r in dm.keys()})

def write_per_subject_csv(path, idx):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "subject"] + all_regions + ["mean_dice"])
        for dataset, subject, dice_mia, dice_sup in results:
            d = (dice_mia, dice_sup)[idx]
            vals = [d.get(r, np.nan) for r in all_regions]
            mean = np.nanmean(vals)
            w.writerow([dataset, subject] + vals + [mean])

write_per_subject_csv(OUTPUT_DIR / "dice_mine_vs_fs.csv", 0)
write_per_subject_csv(OUTPUT_DIR / "dice_jesper_vs_fs.csv", 1)
print(f"CSV per-soggetto -> {OUTPUT_DIR}/dice_mine_vs_fs.csv, dice_jesper_vs_fs.csv")


# ============================================================
# Output 2: media per regione (riassunto)
# ============================================================

with open(OUTPUT_DIR / "dice_per_region_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["region", "mine_mean", "mine_std",
                "supervisor_mean", "supervisor_std", "mine_minus_sup"])
    for r in all_regions:
        mine = np.array([dm.get(r, np.nan) for *_, dm, _ in results], float)
        sup  = np.array([ds.get(r, np.nan) for *_, _, ds in results], float)
        mm, sm = np.nanmean(mine), np.nanmean(sup)
        w.writerow([r, mm, np.nanstd(mine), sm, np.nanstd(sup), mm - sm])
print(f"Riassunto per regione -> {OUTPUT_DIR}/dice_per_region_summary.csv")


# ============================================================
# Output 3: "chi vince" (mean dice per soggetto)
# ============================================================

mine_means = np.array([np.nanmean(list(dm.values())) for *_, dm, _ in results])
sup_means  = np.array([np.nanmean(list(ds.values())) for *_, _, ds in results])

wins_mine = int((mine_means > sup_means).sum())
wins_sup  = int((sup_means > mine_means).sum())
diff = mine_means - sup_means

summary_txt = OUTPUT_DIR / "who_wins.txt"
with open(summary_txt, "w") as f:
    f.write("=== Confronto vs FreeSurfer (ground truth) ===\n\n")
    f.write(f"Soggetti valutati: {len(results)}\n\n")
    f.write(f"Mean Dice (mia)        : {mine_means.mean():.4f} +/- {mine_means.std():.4f}\n")
    f.write(f"Mean Dice (supervisor) : {sup_means.mean():.4f} +/- {sup_means.std():.4f}\n\n")
    f.write(f"Differenza media (mia - sup): {diff.mean():+.4f}\n\n")
    f.write(f"Soggetti dove vinco io       : {wins_mine}/{len(results)}\n")
    f.write(f"Soggetti dove vince supervisor: {wins_sup}/{len(results)}\n")
print(f"Chi vince -> {summary_txt}")

print(f"\n=== RISULTATO ===")
print(f"  Mine        : {mine_means.mean():.4f} +/- {mine_means.std():.4f}")
print(f"  Jesper : {sup_means.mean():.4f} +/- {sup_means.std():.4f}")
print(f"  Differenza : {diff.mean():+.4f}  (vinco {wins_mine}/{len(results)} soggetti)")


# ============================================================
# Output 4: plot
# ============================================================

# (a) Boxplot globale: mean dice per soggetto, mia vs supervisor
fig, ax = plt.subplots(figsize=(7, 6))
ax.boxplot([mine_means, sup_means], labels=["Mine", "Jesper"], showfliers=False)
for j, vals in enumerate([mine_means, sup_means], start=1):
    xs = np.random.normal(j, 0.04, size=len(vals))
    ax.scatter(xs, vals, alpha=0.4, s=20, color="gray")
ax.set_ylabel("Mean Dice vs FreeSurfer")
ax.set_title("Accuratezza vs FreeSurfer: mine vs jesper")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "dice_global_comparison.png", dpi=150)
plt.close(fig)

# (b) Boxplot per regione, affiancati (mia vs supervisor)
fig, ax = plt.subplots(figsize=(max(14, len(all_regions) * 0.8), 6))
positions_mine = np.arange(len(all_regions)) * 2.0
positions_sup  = positions_mine + 0.8
data_mine = [np.array([dm.get(r, np.nan) for *_, dm, _ in results], float) for r in all_regions]
data_sup  = [np.array([ds.get(r, np.nan) for *_, _, ds in results], float) for r in all_regions]
data_mine = [v[~np.isnan(v)] for v in data_mine]
data_sup  = [v[~np.isnan(v)] for v in data_sup]
bp1 = ax.boxplot(data_mine, positions=positions_mine, widths=0.7,
                 patch_artist=True, showfliers=False)
bp2 = ax.boxplot(data_sup, positions=positions_sup, widths=0.7,
                 patch_artist=True, showfliers=False)
for b in bp1["boxes"]:
    b.set_facecolor("#378ADD"); b.set_alpha(0.6)
for b in bp2["boxes"]:
    b.set_facecolor("#D85A30"); b.set_alpha(0.6)
ax.set_xticks(positions_mine + 0.4)
ax.set_xticklabels(all_regions, rotation=90)
ax.set_ylabel("Dice vs FreeSurfer")
ax.set_title("Dice per regione vs FreeSurfer (blu=mine, arancione=jesper)")
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "dice_per_region_comparison.png", dpi=150)
plt.close(fig)

print(f"\nPlot -> {OUTPUT_DIR}")