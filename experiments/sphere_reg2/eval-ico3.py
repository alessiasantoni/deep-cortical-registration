import torch
import numpy as np
import nibabel as nib
from pathlib import Path

import brainsynth.resources
from brainnet.mesh.topology import DeepSurferTopology
from experiments.sphere_reg2.dataset import sample_grid_at_vertices

# SALVATAGGIO SU LAPTOP: SOSTITUISCI CON LA DATA DELLA CARTELLA CHE SALVI
"""
mkdir C:\\spheres_sulc_v2_20260513
scp -r s252125@health-pcalc2:/home/s252125/my_project/experiments/sphere_reg2/evaluation_vtk/20260513_103949/* C:\\spheres_sulc_v2_20260513\\
"""

# ============== CONFIG ==============
SUBJECTS = ["sub-0259", "sub-1297"]
HEMI = "lh"
#DATASET_NAME = "ADHD200"   # nome del dataset come appare nei file salvati
DATASET_NAME = "OASIS3"   # nome del dataset come appare nei file salvati

EPOCH = 50
#SUBSET = "train"   # "train" o "val"

# === LIVELLO ICO della sfera predetta dal modello ===
ICO_LEVEL = 3   # 3, 4, 5, o 6 — DEVE matchare ICO_OUT del training
ORIG_ROOT = Path(f"/projects/brainnet-data/sphere-reg/{DATASET_NAME}")
REG_ROOT = Path(f"/home/s252125/my_project/experiments/sphere_reg2/runs/20260513_103949/spheres/epoch_{EPOCH:03d}")
OUT_DIR = Path("/home/s252125/my_project/experiments/sphere_reg2/evaluation_vtk/20260513_103949")
OUT_DIR.mkdir(parents=True, exist_ok=True)


GRID_PATH = "/home/s252125/my_project/experiments/sphere_reg2/fsaverage_grid/lh.white.features.npz"


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


# ===== Carica template grid =====
grid_data = np.load(GRID_PATH)
template_grid = torch.tensor(grid_data["sulc"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)


# ===== Carica template DeepSurfer al livello ICO corretto =====
print(f"Loading DeepSurfer template at ICO_LEVEL={ICO_LEVEL}")

# Topologie di tutti i livelli da 0 a ICO_LEVEL
topologies = DeepSurferTopology.recursive_subdivision(ICO_LEVEL)
n_verts_level = topologies[ICO_LEVEL].n_vertices
print(f"  Topology has {n_verts_level} vertices")

# Carica sfera template ico6 e prendi solo i primi N vertici (ico_level)
template_resources = brainsynth.resources.Template()
template_verts_full = template_resources.load_surface(HEMI, "sphere.reg")["vertices"].numpy()
template_verts = template_verts_full[:n_verts_level]   # slice ai primi N vertici

# Faces dalla topology al livello giusto
template_faces = topologies[ICO_LEVEL].faces.cpu().numpy().astype(np.int32)

print(f"  template_verts shape: {template_verts.shape}")
print(f"  template_faces shape: {template_faces.shape}")


# ===== TABELLA METRICHE =====
print(f"\n{'Subject':<12} {'MSE template':>13} {'MSE old.reg':>12} {'MSE model':>11} "
      f"{'Δ vs old':>9} {'Δ vs tmpl':>10} {'Mean disp':>10} {'Max disp':>10}")
print("-" * 100)

for sub in SUBJECTS:
    # Carica sulc del soggetto (vertici originali su mesh ico6)
    sulc_subj_full = nib.freesurfer.read_morph_data(str(ORIG_ROOT / sub / f"{HEMI}.sulc"))
    # Slice ai primi N vertici per matchare il livello ICO
    sulc_subj = sulc_subj_full[:n_verts_level]



    # Old sphere.reg (dai dati grezzi, ico6) — slice ai primi N vertici
    verts_old_full, _ = nib.freesurfer.read_geometry(str(ORIG_ROOT / sub / f"{HEMI}.sphere.reg"))
    verts_old = verts_old_full[:n_verts_level]
    faces_old = template_faces   # usiamo le stesse faces (DeepSurferTopology corrispondenti)

    # Sfera predetta dal modello (già al livello ICO_LEVEL)
    verts_model, _ = nib.freesurfer.read_geometry(
        str(REG_ROOT / f"{DATASET_NAME}_{sub}_{HEMI}.sphere.reg")
    )

    # Sanity check
    if verts_model.shape[0] != n_verts_level:
        print(f"  WARNING: {sub} ha {verts_model.shape[0]} vertici, atteso {n_verts_level}")
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

    # ===== SALVA VTK =====

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


# ===== SALVA TEMPLATE da solo =====
sulc_template_only = sample_grid_at_vertices(
    template_grid, torch.from_numpy(template_verts.T).unsqueeze(0).float()
)[0, 0].numpy()

write_vtk(
    OUT_DIR / "template.vtk",
    template_verts, template_faces,
    dict(sulc=normalize(sulc_template_only)),
)

print(f"\nAll saved in: {OUT_DIR}")