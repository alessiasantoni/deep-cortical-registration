"""
Saves as VTK (for ParaView) the spheres AND the white matter of the OUTLIER
subjects, with the aparc label transferred from fsaverage as a scalar field, for:
  - Mine, Jesper, FreeSurfer (per-subject registrations)
  - Initial (fixed, undeformed baseline -- same for all subjects,
    useful for diagnosing whether an outlier already starts out shifted)

Reads the list of outliers produced by compute_parcel_dice.py
(outlier_subjects_mine.csv or outlier_subjects_diff.csv) and recomputes
the labels only for those subjects.

To save the outliers' spheres and wm as vtk
scp -r s252125@health-pcalc2:/home/s252125/my_project/experiments/sphere_reg2/runs/parcel_dice/outlier_vtk/* C:\\outlier_parcel\\

"""

from pathlib import Path
import csv
import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree
import cortech
import brainsynth.resources

HEMI = "lh"
N_VERTS_EXPECTED = 245762

FSAVERAGE_SPHERE = Path("experiments/spherical_registration/fsaverage_aparc/lh.fsaverage.sphere")
FSAVERAGE_ANNOT  = Path("experiments/spherical_registration/fsaverage_aparc/lh.aparc.annot")

PRED_SPHERES_DIR = Path(
    "/home/s252125/my_project/experiments/sphere_reg2/runs/ico6_full_test"
)

DATA_ROOT = Path("/projects/brainnet-data/sphere-reg")
JESPER_SPHERE_NAME = f"{HEMI}.sphere.reg"
FS_SPHERE_NAME     = f"{HEMI}.sphere.reg.freesurfer"
WHITE_NAME         = f"{HEMI}.white"

PARCEL_DIR = Path(
    "/home/s252125/my_project/experiments/sphere_reg2/runs/parcel_dice"
)

OUTLIER_SOURCE = "mine"
OUTLIER_CSV = PARCEL_DIR / f"outlier_subjects_{OUTLIER_SOURCE}.csv"

OUT_DIR = PARCEL_DIR / "outlier_vtk"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_to_unit_sphere(verts):
    return verts / np.linalg.norm(verts, axis=1, keepdims=True)


def transfer_labels(source_sphere, source_labels, target_sphere):
    src = normalize_to_unit_sphere(source_sphere)
    tgt = normalize_to_unit_sphere(target_sphere)
    tree = cKDTree(src)
    _, nn = tree.query(tgt)
    return source_labels[nn]


def save_surface_vtk(out_path, verts, faces, labels):
    s = cortech.Surface(verts, faces)
    s.save(str(out_path), dict(aparc=labels.astype(np.int32)))


# fsaverage upload
print("Loading fsaverage atlas")
fsavg_sphere, _ = nib.freesurfer.read_geometry(str(FSAVERAGE_SPHERE))
fsavg_labels, ctab, region_names = nib.freesurfer.read_annot(str(FSAVERAGE_ANNOT))
assert fsavg_sphere.shape[0] == len(fsavg_labels)



print("Computing FIXED labels on the initial (undeformed) sphere")
template_resources = brainsynth.resources.Template()
template_verts_full = template_resources.load_surface(HEMI, "sphere.reg")["vertices"].numpy()
template_verts = template_verts_full[:N_VERTS_EXPECTED]
labels_initial = transfer_labels(fsavg_sphere, fsavg_labels, template_verts)



if not OUTLIER_CSV.exists():
    raise RuntimeError(f"Outlier file not found: {OUTLIER_CSV}. Run compute_parcel_dice.py first.")

outlier_subjects = []
with open(OUTLIER_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        outlier_subjects.append((row["dataset"], row["subject"]))

print(f"\n{len(outlier_subjects)} outlier subjects to process (source={OUTLIER_SOURCE})")



saved = 0
for dataset, subject in outlier_subjects:
    pred_path   = PRED_SPHERES_DIR / f"{dataset}_{subject}_{HEMI}.sphere.reg"
    sub_dir     = DATA_ROOT / dataset / subject
    jesper_path = sub_dir / JESPER_SPHERE_NAME
    fs_path     = sub_dir / "derivatives" / FS_SPHERE_NAME
    white_path  = sub_dir / WHITE_NAME

    required = (pred_path, jesper_path, fs_path, white_path)
    missing = [p for p in required if not p.exists()]
    if missing:
        print(f"  [{dataset}/{subject}] missing files: {[str(m) for m in missing]}, skip")
        continue

    mine_verts,   mine_faces   = nib.freesurfer.read_geometry(str(pred_path))
    jesper_verts, jesper_faces = nib.freesurfer.read_geometry(str(jesper_path))
    fs_verts,     fs_faces     = nib.freesurfer.read_geometry(str(fs_path))
    white_verts,  white_faces  = nib.freesurfer.read_geometry(str(white_path))

    if white_verts.shape[0] != mine_verts.shape[0]:
        print(f"  [{dataset}/{subject}] WARNING: white has {white_verts.shape[0]} vertices, "
              f"sphere has {mine_verts.shape[0]} — skipping white matter for this subject")
        white_ok = False
    else:
        white_ok = True

    labels_mine   = transfer_labels(fsavg_sphere, fsavg_labels, mine_verts)
    labels_jesper = transfer_labels(fsavg_sphere, fsavg_labels, jesper_verts)
    labels_fs     = transfer_labels(fsavg_sphere, fsavg_labels, fs_verts)
    # labels_initial is already computed once outside the loop (fixed)

    prefix = f"{dataset}_{subject}"

    # --- Spheres ---
    save_surface_vtk(OUT_DIR / f"{prefix}_mine_sphere.vtk",    mine_verts,    mine_faces,    labels_mine)
    save_surface_vtk(OUT_DIR / f"{prefix}_jesper_sphere.vtk",  jesper_verts,  jesper_faces,  labels_jesper)
    save_surface_vtk(OUT_DIR / f"{prefix}_fs_sphere.vtk",      fs_verts,      fs_faces,      labels_fs)
    save_surface_vtk(OUT_DIR / f"{prefix}_initial_sphere.vtk", template_verts, mine_faces,   labels_initial)

    if white_ok:
        save_surface_vtk(OUT_DIR / f"{prefix}_mine_wm.vtk",    white_verts, white_faces, labels_mine)
        save_surface_vtk(OUT_DIR / f"{prefix}_jesper_wm.vtk",  white_verts, white_faces, labels_jesper)
        save_surface_vtk(OUT_DIR / f"{prefix}_fs_wm.vtk",      white_verts, white_faces, labels_fs)
        save_surface_vtk(OUT_DIR / f"{prefix}_initial_wm.vtk", white_verts, white_faces, labels_initial)

    saved += 1
    n_files = 8 if white_ok else 4
    print(f"  [{saved}/{len(outlier_subjects)}] {prefix}: saved {n_files} VTK files")

print(f"\nDone. {saved} subjects saved to: {OUT_DIR}")
print("Per subject: _mine_sphere.vtk, _jesper_sphere.vtk, _fs_sphere.vtk, _initial_sphere.vtk")
print("            + _mine_wm.vtk, _jesper_wm.vtk, _fs_wm.vtk, _initial_wm.vtk")