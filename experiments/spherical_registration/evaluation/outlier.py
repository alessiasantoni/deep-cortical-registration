"""
Saves as VTK (for ParaView) the spheres AND the white matter of the OUTLIER
subjects, with the aparc label transferred from fsaverage as a scalar field, for:
  - Mine, JOSA, FreeSurfer (per-subject registrations)
  - Initial (fixed, undeformed baseline -- same for all subjects,
    useful for diagnosing whether an outlier already starts out shifted)

Reads the list of outliers produced by parcel_dice.py
(<run_dir>/parcel_dice/outlier_subjects_mine.csv) and recomputes
the labels only for those subjects.

Usage (from the repository root, after parcel_dice.py):
  python -m experiments.spherical_registration.evaluation.outlier --run_dir runs/ico6_train
Writes <run_dir>/parcel_dice/outlier_vtk/.
"""

from pathlib import Path
import argparse
import csv
import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree
import cortech
import brainsynth.resources

HEMI = "lh"
N_VERTS_EXPECTED = 245762

# fsaverage aparc resampled on the brainsynth template mesh: its labels are
# indexed like the vertices of the template sphere.reg, so it must be paired with that sphere
FSAVERAGE_ANNOT = Path(f"experiments/spherical_registration/fsaverage_aparc/{HEMI}.aparc.annot")

DATA_ROOT = Path("/projects/brainnet-data/sphere-reg")
JOSA_SPHERE_NAME   = f"{HEMI}.sphere.reg.josa"
FS_SPHERE_NAME     = f"{HEMI}.sphere.reg.freesurfer"
WHITE_NAME         = f"{HEMI}.white"

parser = argparse.ArgumentParser(description="Save the outlier subjects' surfaces as VTK, with aparc labels.")
parser.add_argument("--run_dir", default="runs/ico6_train",
                    help="run folder containing parcel_dice/ (written by parcel_dice.py)")
parser.add_argument("--pred_dir", default=None,
                    help="folder with the predicted spheres (default: <run_dir>/test_predictions)")
args = parser.parse_args()

PRED_SPHERES_DIR = Path(args.pred_dir) if args.pred_dir else Path(args.run_dir) / "test_predictions"
PARCEL_DIR = Path(args.run_dir) / "parcel_dice"

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
template_resources = brainsynth.resources.Template()
fsavg_sphere = template_resources.load_surface(HEMI, "sphere.reg")["vertices"].numpy()
fsavg_labels, ctab, region_names = nib.freesurfer.read_annot(str(FSAVERAGE_ANNOT))
assert fsavg_sphere.shape[0] == len(fsavg_labels)



print("Computing FIXED labels on the initial (undeformed) sphere")
template_verts = fsavg_sphere[:N_VERTS_EXPECTED]
labels_initial = transfer_labels(fsavg_sphere, fsavg_labels, template_verts)



if not OUTLIER_CSV.exists():
    raise RuntimeError(f"Outlier file not found: {OUTLIER_CSV}. Run parcel_dice.py first.")

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
    josa_path   = sub_dir / "derivatives" / JOSA_SPHERE_NAME
    fs_path     = sub_dir / "derivatives" / FS_SPHERE_NAME
    white_path  = sub_dir / WHITE_NAME

    required = (pred_path, josa_path, fs_path, white_path)
    missing = [p for p in required if not p.exists()]
    if missing:
        print(f"  [{dataset}/{subject}] missing files: {[str(m) for m in missing]}, skip")
        continue

    mine_verts,   mine_faces   = nib.freesurfer.read_geometry(str(pred_path))
    josa_verts, josa_faces = nib.freesurfer.read_geometry(str(josa_path))
    fs_verts,     fs_faces     = nib.freesurfer.read_geometry(str(fs_path))
    white_verts,  white_faces  = nib.freesurfer.read_geometry(str(white_path))

    if white_verts.shape[0] != mine_verts.shape[0]:
        print(f"  [{dataset}/{subject}] WARNING: white has {white_verts.shape[0]} vertices, "
              f"sphere has {mine_verts.shape[0]} — skipping white matter for this subject")
        white_ok = False
    else:
        white_ok = True

    labels_mine   = transfer_labels(fsavg_sphere, fsavg_labels, mine_verts)
    labels_josa = transfer_labels(fsavg_sphere, fsavg_labels, josa_verts)
    labels_fs     = transfer_labels(fsavg_sphere, fsavg_labels, fs_verts)
    # labels_initial is already computed once outside the loop (fixed)

    prefix = f"{dataset}_{subject}"

    # --- Spheres ---
    save_surface_vtk(OUT_DIR / f"{prefix}_mine_sphere.vtk",    mine_verts,    mine_faces,    labels_mine)
    save_surface_vtk(OUT_DIR / f"{prefix}_josa_sphere.vtk",  josa_verts,  josa_faces,  labels_josa)
    save_surface_vtk(OUT_DIR / f"{prefix}_fs_sphere.vtk",      fs_verts,      fs_faces,      labels_fs)
    save_surface_vtk(OUT_DIR / f"{prefix}_initial_sphere.vtk", template_verts, mine_faces,   labels_initial)

    if white_ok:
        save_surface_vtk(OUT_DIR / f"{prefix}_mine_wm.vtk",    white_verts, white_faces, labels_mine)
        save_surface_vtk(OUT_DIR / f"{prefix}_josa_wm.vtk",  white_verts, white_faces, labels_josa)
        save_surface_vtk(OUT_DIR / f"{prefix}_fs_wm.vtk",      white_verts, white_faces, labels_fs)
        save_surface_vtk(OUT_DIR / f"{prefix}_initial_wm.vtk", white_verts, white_faces, labels_initial)

    saved += 1
    n_files = 8 if white_ok else 4
    print(f"  [{saved}/{len(outlier_subjects)}] {prefix}: saved {n_files} VTK files")

print(f"\nDone. {saved} subjects saved to: {OUT_DIR}")
print("Per subject: _mine_sphere.vtk, _josa_sphere.vtk, _fs_sphere.vtk, _initial_sphere.vtk")
print("            + _mine_wm.vtk, _josa_wm.vtk, _fs_wm.vtk, _initial_wm.vtk")