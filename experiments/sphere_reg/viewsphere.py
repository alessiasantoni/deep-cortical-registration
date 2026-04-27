import cortech
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import brainsynth.resources
from pathlib import Path

# Path
spheres_dir = Path("/home/s252125/my_project/experiments/sphere_reg/runs/sulcal_reg/spheres/epoch_030")
sulc_root = Path("/projects/brainnet-data/sphere-reg/Buckner40")
out_dir = Path("/home/s252125/my_project/experiments/sphere_reg/vtk_output")
out_dir.mkdir(exist_ok=True)

# === SOGGETTI ===
subjects = ["sub-001", "sub-017", "sub-019", "sub-037", "sub-039"]

for sub in subjects:
    sphere_path = spheres_dir / f"Buckner40_{sub}_lh.sphere.reg"
    sulc_path = sulc_root / sub / "lh.sulc"
    
    s = cortech.Surface.from_file(str(sphere_path))
    sulc = nib.freesurfer.read_morph_data(str(sulc_path))
    
    out_path = out_dir / f"{sub}_lh.vtk"
    s.save(str(out_path), scalars=dict(sulc=sulc))
    print(f"Saved: {out_path}")


# === TEMPLATE ===
template = brainsynth.resources.Template()
template_verts = template.load_surface("lh", "sphere.reg")["vertices"]
template_faces = template.load_surface("lh", "sphere.reg")["faces"].numpy()

# Carica la griglia 2D del template
grid_data = np.load("/home/s252125/my_project/experiments/sphere_reg/fsaverage_grid/lh.white.features.npz")
sulc_grid = torch.tensor(grid_data["sulc"], dtype=torch.float32)

# Cartesiane → sferiche → coordinate griglia
v_unit = F.normalize(template_verts.float(), dim=-1)
theta = torch.arccos(v_unit[:, 2].clamp(-1, 1))
phi = torch.arctan2(v_unit[:, 1], v_unit[:, 0])

grid_x = phi / torch.pi
grid_y = (theta / torch.pi) * 2.0 - 1.0

sampling_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(2)
sampled = F.grid_sample(
    sulc_grid.unsqueeze(0).unsqueeze(0),
    sampling_grid,
    mode="bilinear",
    padding_mode="border",
    align_corners=True,
)
template_sulc = sampled.squeeze().numpy()

# Salva il template con sulc
import pyvista as pv
faces_pv = np.hstack([np.full((len(template_faces), 1), 3), template_faces]).astype(np.int64)
mesh = pv.PolyData(template_verts.numpy(), faces_pv)
mesh.point_data["sulc"] = template_sulc
out_path = out_dir / "template_lh.vtk"
mesh.save(str(out_path))
print(f"Saved: {out_path}")




# Il seguente comando scritto nel terminale salva le sfere e i sulc + template e sulc del template in formato vtk
# nella cartella questoPC->spheres_sulc

# scp -r s252125@health-pcalc2:/home/s252125/my_project/experiments/sphere_reg/vtk_output/* C:\spheres_sulc\vtk\