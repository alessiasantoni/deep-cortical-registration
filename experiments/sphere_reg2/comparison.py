import torch
import numpy as np
from pathlib import Path
import nibabel as nib

from experiments.sphere_reg2.dataset import sample_grid_at_vertices


# ================= CONFIG =================
SUBJECT = "sub-0043"
HEMI = "lh"

SPHERE_ORIG = Path(
    f"/projects/brainnet-data/sphere-reg/ABIDE/{SUBJECT}/{HEMI}.sphere.reg"
)
SPHERE_REG  = Path(f"/home/s252125/my_project/experiments/sphere_reg2/runs/spheres/epoch_030/ABIDE_{SUBJECT}_{HEMI}.sphere.reg")

SULC_SUBJ_PATH = Path(f"/projects/brainnet-data/sphere-reg/ABIDE/{SUBJECT}/{HEMI}.sulc")


OUT_DIR = Path("/home/s252125/my_project/debug_vtk")
OUT_DIR.mkdir(exist_ok=True)
# ==========================================


def write_vtk(path, verts, faces, scalars: dict):
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("sphere\nASCII\nDATASET POLYDATA\n")

        f.write(f"POINTS {len(verts)} float\n")
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

        f.write(f"POLYGONS {len(faces)} {len(faces)*4}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

        f.write(f"\nPOINT_DATA {len(verts)}\n")

        for name, data in scalars.items():
            data = np.asarray(data).reshape(-1)
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for v in data:
                f.write(f"{float(v)}\n")


# ===== LOAD DATA =====
verts_orig, faces = nib.freesurfer.read_geometry(str(SPHERE_ORIG))
verts_reg, _      = nib.freesurfer.read_geometry(str(SPHERE_REG))

sulc_subj = nib.freesurfer.read_morph_data(str(SULC_SUBJ_PATH))

grid_data = np.load("/home/s252125/my_project/experiments/sphere_reg/fsaverage_grid/lh.white.features.npz")
sulc_grid = torch.tensor(grid_data["sulc"], dtype=torch.float32)

template_grid = sulc_grid.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)


# ===== TORCH FORMAT =====
verts_orig_t = torch.from_numpy(verts_orig.T).unsqueeze(0).float()  # (1, 3, N)
verts_reg_t  = torch.from_numpy(verts_reg.T).unsqueeze(0).float()

template_grid = template_grid.unsqueeze(0) if template_grid.ndim == 3 else template_grid


# ===== SAMPLE TEMPLATE =====
sulc_temp_orig = sample_grid_at_vertices(template_grid, verts_orig_t)[0, 0].cpu().numpy()
sulc_temp_reg  = sample_grid_at_vertices(template_grid, verts_reg_t)[0, 0].cpu().numpy()


# ===== DIFFERENCES =====
diff_before = sulc_subj - sulc_temp_orig
diff_after  = sulc_subj - sulc_temp_reg


# ===== SAVE =====

# 1 + 2 (original sphere)
write_vtk(
    OUT_DIR / f"{SUBJECT}_orig.vtk",
    verts_orig,
    faces,
    dict(
        sulc_subj=sulc_subj,
        sulc_template=sulc_temp_orig,
        diff=diff_before
    )
)

# 3 + 4 (registered sphere)
write_vtk(
    OUT_DIR / f"{SUBJECT}_reg.vtk",
    verts_reg,
    faces,
    dict(
        sulc_subj=sulc_subj,
        sulc_template=sulc_temp_reg,
        diff=diff_after
    )
)

print("Saved in:", OUT_DIR)