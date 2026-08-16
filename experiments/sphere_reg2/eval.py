"""
This code works as eval-ico3 but it detects automatically the ico_level of 
the predicted sphere, so that it cam be used with any ICO_LEVEL without hardcoding it.
"""

import torch
import numpy as np
import nibabel as nib
from pathlib import Path

import brainsynth.resources
from brainnet.mesh.topology import DeepSurferTopology
from experiments.sphere_reg2.dataset import sample_grid_at_vertices

"""
mkdir C:\\spheres_sulc_v2_20260520
scp -r s252125@health-pcalc2:/home/s252125/my_project/experiments/sphere_reg2/evaluation_vtk/20260609_181249_finetune_ico6/* C:\\worst_sub_parcellation\\
"""

# ============== CONFIG ==============
SUBJECTS = ["sub-1210", "sub-0807", "sub-0076", "sub-1075", "sub-1039", "sub-0715", "sub-0731", "sub-0805", "sub-0304"]
HEMI = "lh"
DATASET_NAME = "OASIS3"


EPOCH = 20

# NOTA: ICO_LEVEL non e' piu' hardcoded - viene dedotto dalla sfera predetta.
# Il numero massimo per la ricerca della topology corretta:
MAX_ICO_LEVEL = 6

ORIG_ROOT = Path(f"/projects/brainnet-data/sphere-reg/{DATASET_NAME}")
REG_ROOT = Path(
    "/home/s252125/my_project/experiments/sphere_reg2/runs/"
    f"20260609_181249_finetune_ico6/spheres_all"
)
#REG_ROOT = Path(f"20260607_213421_finetune_ico4_frozen/spheres/epoch_{EPOCH:03d}")
    #f"/home/s252125/my_project/experiments/sphere_reg2/runs/"
    #f"20260607_213421_finetune_ico4_frozen/spheres"

#    f"20260607_143539_finetune_ico4_frozen/spheres_all_{DATASET_NAME}/epoch_{EPOCH:03d}"

OUT_DIR = Path(
    "/home/s252125/my_project/experiments/sphere_reg2/evaluation_vtk/"
    "20260609_181249_finetune_ico6"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_PATH = "/home/s252125/my_project/experiments/sphere_reg2/fsaverage_grid/lh.white.features.npz"
# ====================================


# ============================================================
# Utilities
# ============================================================

def write_vtk(path, verts, faces, scalars: dict):
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nsphere\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {len(verts)} float\n")
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        f.write(f"POLYGONS {len(faces)} {len(faces)*4}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        f.write(f"\nPOINT_DATA {len(verts)}\n")
        for name, data in scalars.items():
            data = np.asarray(data).reshape(-1)
            f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
            for v in data:
                f.write(f"{float(v)}\n")


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def compute_mse(a, b):
    a_n = normalize(a)
    b_n = normalize(b)
    return float(((a_n - b_n) ** 2).mean())


def detect_ico_level(n_verts: int, topologies) -> int:
    """Trova il livello ICO che ha esattamente n_verts vertici.
    
    Lancia eccezione se nessun livello corrisponde.
    """
    for level in range(len(topologies)):
        if topologies[level].n_vertices == n_verts:
            return level
    available = [t.n_vertices for t in topologies]
    raise ValueError(
        f"Nessun livello ICO ha {n_verts} vertici. "
        f"Disponibili: {available}"
    )


# ============================================================
# Carica risorse (una volta)
# ============================================================

print(f"Loading template grid from {GRID_PATH}")
grid_data = np.load(GRID_PATH)
template_grid = torch.tensor(grid_data["sulc"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

print(f"Loading DeepSurfer topologies up to ICO={MAX_ICO_LEVEL}")
topologies = DeepSurferTopology.recursive_subdivision(MAX_ICO_LEVEL)
print("  Available topology levels:")
for level in range(len(topologies)):
    print(f"    ico={level}: {topologies[level].n_vertices} vertices, "
          f"{topologies[level].faces.shape[0]} faces")

# Sfera template fsaverage (alla risoluzione massima, slicabile)
template_resources = brainsynth.resources.Template()
template_verts_full = template_resources.load_surface(HEMI, "sphere.reg")["vertices"].numpy()
print(f"  Template fsaverage sphere has {template_verts_full.shape[0]} vertices")


# ============================================================
# Auto-detect ICO_LEVEL dal primo soggetto disponibile
# ============================================================

print(f"\nAuto-detecting ICO_LEVEL from predicted spheres in {REG_ROOT}")
first_sphere_path = None
for sub in SUBJECTS:
    candidate = REG_ROOT / f"{DATASET_NAME}_{sub}_{HEMI}.sphere.reg"
    if candidate.exists():
        first_sphere_path = candidate
        break

if first_sphere_path is None:
    raise RuntimeError(f"No predicted spheres found in {REG_ROOT} for any subject in {SUBJECTS}")

probe_verts, _ = nib.freesurfer.read_geometry(str(first_sphere_path))
n_verts_level = probe_verts.shape[0]
ICO_LEVEL = detect_ico_level(n_verts_level, topologies)
print(f"  Detected: ICO_LEVEL = {ICO_LEVEL} ({n_verts_level} vertices)")

# Ora usiamo questi per tutto il resto
template_verts = template_verts_full[:n_verts_level]
template_faces = topologies[ICO_LEVEL].faces.cpu().numpy().astype(np.int32)

print(f"  template_verts shape: {template_verts.shape}")
print(f"  template_faces shape: {template_faces.shape}")


# ============================================================
# Tabella metriche
# ============================================================

print(f"\n{'Subject':<12} {'MSE template':>13} {'MSE old.reg':>12} {'MSE model':>11} "
      f"{'Δ vs old':>9} {'Δ vs tmpl':>10} {'Mean disp':>10} {'Max disp':>10}")
print("-" * 100)

for sub in SUBJECTS:
    # Sulc del soggetto (mesh nativa, slice ai primi n_verts_level)
    sulc_subj_full = nib.freesurfer.read_morph_data(str(ORIG_ROOT / sub / "derivatives" / f"{HEMI}.sulc"))
    sulc_subj = sulc_subj_full[:n_verts_level]

    # Old sphere.reg di FreeSurfer (slice ai primi n_verts_level)
    verts_old_full, _ = nib.freesurfer.read_geometry(
        str(ORIG_ROOT / sub / f"{HEMI}.sphere.reg")
    )
    verts_old = verts_old_full[:n_verts_level]
    faces_old = template_faces

    # Sfera predetta dal modello (gia' al livello ICO_LEVEL)
    model_sphere_path = REG_ROOT / f"{DATASET_NAME}_{sub}_{HEMI}.sphere.reg"
    if not model_sphere_path.exists():
        print(f"  {sub}: predicted sphere not found, skip")
        continue

    verts_model, _ = nib.freesurfer.read_geometry(str(model_sphere_path))

    if verts_model.shape[0] != n_verts_level:
        print(f"  WARNING: {sub} ha {verts_model.shape[0]} vertici, atteso {n_verts_level} — skip")
        continue

    # Sample template alle 3 posizioni
    v_template_t = torch.from_numpy(template_verts.T).unsqueeze(0).float()
    v_old_t      = torch.from_numpy(verts_old.T).unsqueeze(0).float()
    v_model_t    = torch.from_numpy(verts_model.T).unsqueeze(0).float()

    sulc_t_template = sample_grid_at_vertices(template_grid, v_template_t)[0, 0].numpy()
    sulc_t_old      = sample_grid_at_vertices(template_grid, v_old_t)[0, 0].numpy()
    sulc_t_model    = sample_grid_at_vertices(template_grid, v_model_t)[0, 0].numpy()

    # Differenze
    diff_template = sulc_subj - sulc_t_template
    diff_old      = sulc_subj - sulc_t_old
    diff_model    = sulc_subj - sulc_t_model

    # MSE
    mse_template = compute_mse(sulc_subj, sulc_t_template)
    mse_old      = compute_mse(sulc_subj, sulc_t_old)
    mse_model    = compute_mse(sulc_subj, sulc_t_model)

    # Displacement
    disp = np.linalg.norm(template_verts - verts_model, axis=1)

    print(f"{sub:<12} {mse_template:>13.4f} {mse_old:>12.4f} {mse_model:>11.4f} "
          f"{mse_model-mse_old:>+9.4f} {mse_model-mse_template:>+10.4f} "
          f"{disp.mean():>10.3f} {disp.max():>10.3f}")

    # ===== Salva VTK =====
    write_vtk(
        OUT_DIR / f"{sub}_init_template.vtk",
        template_verts, template_faces,
        dict(
            sulc_subj=normalize(sulc_subj),
            sulc_template=normalize(sulc_t_template),
            diff=diff_template,
        ),
    )

    write_vtk(
        OUT_DIR / f"{sub}_old_spherereg.vtk",
        verts_old, faces_old,
        dict(
            sulc_subj=normalize(sulc_subj),
            sulc_template=normalize(sulc_t_old),
            diff=diff_old,
        ),
    )

    write_vtk(
        OUT_DIR / f"{sub}_model.vtk",
        verts_model, template_faces,
        dict(
            sulc_subj=normalize(sulc_subj),
            sulc_template=normalize(sulc_t_model),
            diff=diff_model,
        ),
    )


# ===== Salva template da solo =====
sulc_template_only = sample_grid_at_vertices(
    template_grid, torch.from_numpy(template_verts.T).unsqueeze(0).float()
)[0, 0].numpy()

write_vtk(
    OUT_DIR / "template.vtk",
    template_verts, template_faces,
    dict(sulc=normalize(sulc_template_only)),
)

print(f"\nAll saved in: {OUT_DIR}")