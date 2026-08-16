"""
Dice before/after registration + parcellation visualization on the white matter.

"before" = atlas transferred via the undeformed fsaverage template sphere
           (same for all subjects: the starting point of the network).
"after"  = atlas transferred via my registered sphere (.sphere.reg).
Jesper   = atlas transferred via lh.sphere.reg (subject root).
FS       = atlas transferred via derivatives/lh.sphere.reg.freesurfer (ground truth).

Global Dice with FreeSurfer as ground truth, plus parcellation painted on the
subject's white matter (same geometry, different label colors) saved as VTK.

scp -r s252125@health-pcalc2:/home/s252125/my_project/experiments/sphere_reg2/eval_before_after/* C:\\worst_parcellation\\

"""

from pathlib import Path
import csv
import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree
import brainsynth.resources


# ============================================================
# CONFIG
# ============================================================

HEMI = "lh"
DATASET_NAME = "COBRE"

#SUBJECTS = ["sub-1210", "sub-0076", "sub-0807", "sub-1075", "sub-0304"]
SUBJECTS = ["sub-123", "sub-040", "sub-016", "sub-019", "sub-100", "sub-169", "sub-066", "sub-036", "sub-027", "sub-148",]

FSAVERAGE_SPHERE = Path("experiments/sphere_reg2/fsaverage_aparc/lh.fsaverage.sphere")
FSAVERAGE_ANNOT  = Path("experiments/sphere_reg2/fsaverage_aparc/lh.aparc.annot")

PRED_SPHERES_DIR = Path(
    "/home/s252125/my_project/experiments/sphere_reg2/runs/"
    "20260609_181249_finetune_ico6/spheres_all"
)

DATA_ROOT = Path(f"/projects/brainnet-data/sphere-reg/{DATASET_NAME}")

OUTPUT_DIR = Path("experiments/sphere_reg2/eval_before_after")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_VERTS_EXPECTED = 245762   # ico=6


# ============================================================
# Helpers
# ============================================================

def normalize_to_unit_sphere(verts):
    return verts / np.linalg.norm(verts, axis=1, keepdims=True)


def transfer_labels(source_sphere, source_labels, target_sphere):
    src = normalize_to_unit_sphere(source_sphere)
    tgt = normalize_to_unit_sphere(target_sphere)
    tree = cKDTree(src)
    _, nn = tree.query(tgt)
    return source_labels[nn]


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

def write_freesurfer_surface(path, verts, faces):
    """Save mesh as FreeSurfer binary (loadable by FreeView with -f)."""
    nib.freesurfer.write_geometry(str(path), verts, faces)


def write_freesurfer_annot(path, labels, ctab, region_names):
    """Save parcellation as FreeSurfer .annot (uses standard colors in FreeView)."""
    # nibabel expects names as bytes
    names_bytes = [n if isinstance(n, bytes) else n.encode("utf-8") for n in region_names]
    nib.freesurfer.write_annot(
        str(path),
        labels.astype(np.int32),
        ctab.astype(np.int32),
        names_bytes,
        fill_ctab=False,
    )


# ============================================================
# Load fsaverage atlas (once) + sanity checks
# ============================================================

print("Loading fsaverage atlas")
fsavg_sphere, _ = nib.freesurfer.read_geometry(str(FSAVERAGE_SPHERE))
fsavg_labels, ctab, region_names = nib.freesurfer.read_annot(str(FSAVERAGE_ANNOT))
print(f"  sphere: {fsavg_sphere.shape[0]} vertices, "
      f"annot: {len(fsavg_labels)} vertices, {len(region_names)} regions")

assert fsavg_sphere.shape[0] == len(fsavg_labels), (
    f"fsaverage MISMATCH: sphere {fsavg_sphere.shape[0]} vs annot {len(fsavg_labels)}"
)

labels_self = transfer_labels(fsavg_sphere, fsavg_labels, fsavg_sphere)
self_match = (labels_self == fsavg_labels).mean()
print(f"  self-transfer sanity: {self_match:.4f} (must be ~1.0)")
assert self_match > 0.99, f"Self-transfer {self_match:.4f} != 1.0 — wrong NN direction?"


# ============================================================
# "BEFORE" state: atlas on the undeformed template sphere (shared by all)
# ============================================================

template_resources = brainsynth.resources.Template()
template_sphere_before = (
    template_resources.load_surface(HEMI, "sphere.reg")["vertices"]
    .numpy()[:N_VERTS_EXPECTED]
)
print(f"Template sphere (BEFORE state): {template_sphere_before.shape}")

# Computed once — same for every subject
labels_before_shared = transfer_labels(fsavg_sphere, fsavg_labels, template_sphere_before)


# ============================================================
# Per-subject processing
# ============================================================

def process_subject(sub: str):
    sub_dir = DATA_ROOT / sub
    after_path  = PRED_SPHERES_DIR / f"{DATASET_NAME}_{sub}_{HEMI}.sphere.reg"
    jesper_path = sub_dir / f"{HEMI}.sphere.reg"
    fs_path     = sub_dir / "derivatives" / f"{HEMI}.sphere.reg.freesurfer"
    white_path  = sub_dir / f"{HEMI}.white"

    needed = {"after": after_path, "Jesper": jesper_path,
              "FS": fs_path, "white": white_path}
    for name, p in needed.items():
        if not p.exists():
            print(f"  [{sub}] missing {name}: {p}, skip")
            return None

    after_sphere,  _ = nib.freesurfer.read_geometry(str(after_path))
    jesper_sphere, _ = nib.freesurfer.read_geometry(str(jesper_path))
    fs_sphere,     _ = nib.freesurfer.read_geometry(str(fs_path))
    white_verts, white_faces = nib.freesurfer.read_geometry(str(white_path))

    for name, s in [("after", after_sphere), ("Jesper", jesper_sphere),
                    ("FS", fs_sphere), ("white", white_verts)]:
        if s.shape[0] != N_VERTS_EXPECTED:
            print(f"  [{sub}] {name} has {s.shape[0]} verts, skip")
            return None

    labels_before = labels_before_shared                                  # prima (condivisa)
    labels_after  = transfer_labels(fsavg_sphere, fsavg_labels, after_sphere)
    labels_jesper = transfer_labels(fsavg_sphere, fsavg_labels, jesper_sphere)
    labels_fs     = transfer_labels(fsavg_sphere, fsavg_labels, fs_sphere)

    d_before = global_dice(labels_before, labels_fs)
    d_after  = global_dice(labels_after,  labels_fs)
    d_jesper = global_dice(labels_jesper, labels_fs)

    print(f"[{sub}]  before: {d_before:.4f} -> after: {d_after:.4f} "
          f"(improvement {d_after - d_before:+.4f}) | Jesper: {d_jesper:.4f}")

    # Save white matter surface once (geometria condivisa per i 3 annot)
    white_out = OUTPUT_DIR / f"{sub}_{HEMI}.white"
    write_freesurfer_surface(white_out, white_verts, white_faces)

    # Save three annot files (same geometry, different labels)
    write_freesurfer_annot(
        OUTPUT_DIR / f"{sub}_{HEMI}.parcellation_before.annot",
        labels_before, ctab, region_names,
    )
    write_freesurfer_annot(
        OUTPUT_DIR / f"{sub}_{HEMI}.parcellation_after.annot",
        labels_after, ctab, region_names,
    )
    write_freesurfer_annot(
        OUTPUT_DIR / f"{sub}_{HEMI}.parcellation_fs.annot",
        labels_fs, ctab, region_names,
    )
    
    return sub, d_before, d_after, d_jesper


# ============================================================
# Loop
# ============================================================

results = []
for sub in SUBJECTS:
    r = process_subject(sub)
    if r is not None:
        results.append(r)

if not results:
    raise RuntimeError("No subjects processed. Check paths and vertex counts.")


# ============================================================
# Output: CSV + summary
# ============================================================

csv_path = OUTPUT_DIR / "dice_before_after.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["subject", "dice_before", "dice_after", "improvement", "dice_jesper"])
    for sub, db, da, dj in results:
        w.writerow([sub, f"{db:.4f}", f"{da:.4f}", f"{da - db:+.4f}", f"{dj:.4f}"])
print(f"\nCSV -> {csv_path}")

before = np.array([db for _, db, _, _ in results])
after  = np.array([da for _, _, da, _ in results])
jesper = np.array([dj for _, _, _, dj in results])
impr   = after - before

summary = OUTPUT_DIR / "before_after_summary.txt"
with open(summary, "w") as f:
    f.write("=== Dice before vs after registration (global, vs FreeSurfer) ===\n\n")
    f.write(f"Subjects: {len(results)}\n\n")
    f.write(f"Mean Dice BEFORE : {before.mean():.4f} +/- {before.std():.4f}\n")
    f.write(f"Mean Dice AFTER  : {after.mean():.4f} +/- {after.std():.4f}\n")
    f.write(f"Mean improvement : {impr.mean():+.4f}\n\n")
    f.write(f"Mean Dice Jesper : {jesper.mean():.4f} +/- {jesper.std():.4f}\n")
print(f"Summary -> {summary}")

print("\n=== RESULT ===")
print(f"  BEFORE : {before.mean():.4f}")
print(f"  AFTER  : {after.mean():.4f}  (improvement {impr.mean():+.4f})")
print(f"  Jesper : {jesper.mean():.4f}")
print(f"\nVTK parcellations -> {OUTPUT_DIR}")