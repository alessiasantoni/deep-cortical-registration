"""
In this script we compare the old registered sphere (from Jesper) and the new model sphere for a few subjects.
I compute the MSE between the template sulc sampled on the spheres and the subject sulc, 
as well as the correlation between the the template sulc sampled on the spheres and the subject sulc on them.
Then I also compute the number of folded triangles in the old and new spheres.

"""

from pathlib import Path
import csv

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from brainnet.mesh.topology import DeepSurferTopology
from experiments.sphere_reg2.dataset_wm import (
    load_template_grid,
    sample_grid_at_vertices,
)

# ============== CONFIG ==============
SUBJECTS = ["sub-003", "sub-016", "sub-021", "sub-023", "sub-026"]

HEMI = "lh"
DATASET_NAME = "Buckner40"
EPOCH = 50
ICO_LEVEL = 3

SPHERE_RADIUS = 100.0

RUN_DIR = Path("/home/s252125/my_project/experiments/sphere_reg2/runs/20260513_162935")
#RUN_DIR = Path("/home/s252125/my_project/experiments/sphere_reg2/ablation_study/nsteps_20_20260513_154859")  # --- IGNORE, example of ablation dir ---
ORIG_ROOT = Path(
    f"/projects/brainnet-data/sphere-reg/{DATASET_NAME}"
)

REG_ROOT = RUN_DIR / "spheres" / f"epoch_{EPOCH:03d}"

GRID_DIR = Path(
    "/home/s252125/my_project/experiments/sphere_reg2/fsaverage_grid"
)

OUT_CSV = RUN_DIR / "comparison_old_vs_new.csv"

TEMPLATE_SULC_STATS = {
    "lh": {"mean": -1.1523885e-08, "std": 0.59384775},
    "rh": {"mean": -4.1938106e-09, "std": 0.5924748},
}
# ====================================


def normalize_sulc(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten()
    b = b.flatten()

    a = a - a.mean()
    b = b - b.mean()

    corr = (a * b).sum() / (
        torch.sqrt((a ** 2).sum()) *
        torch.sqrt((b ** 2).sum()) +
        1e-8
    )

    return corr.item()


def count_folded_triangles(verts: torch.Tensor, faces: torch.Tensor):
    """
    verts: (1, 3, N)
    faces: (F, 3)

    Returns:
        n_folded, perc_folded
    """
    verts = verts.transpose(1, 2)  # (1, N, 3)

    v0 = verts[:, faces[:, 0]]
    v1 = verts[:, faces[:, 1]]
    v2 = verts[:, faces[:, 2]]

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    normals_unit = normals / (
        normals.norm(dim=-1, keepdim=True) + 1e-8
    )

    bary = (v0 + v1 + v2) / 3.0
    bary_unit = bary / (
        bary.norm(dim=-1, keepdim=True) + 1e-8
    )

    d = (normals_unit * bary_unit).sum(dim=-1)

    folded = d < 0
    n_folded = folded.sum().item()
    perc_folded = 100.0 * n_folded / faces.shape[0]

    return n_folded, perc_folded


def main():
    print(f"REG_ROOT = {REG_ROOT}")
    print(f"OUT_CSV  = {OUT_CSV}")

    if not REG_ROOT.exists():
        raise RuntimeError(f"REG_ROOT does not exist: {REG_ROOT}")

    # ===== Topology =====
    print(f"\nLoading topology at ICO_LEVEL={ICO_LEVEL}")
    topologies = DeepSurferTopology.recursive_subdivision(ICO_LEVEL)

    faces = topologies[ICO_LEVEL].faces.long()
    n_verts_level = topologies[ICO_LEVEL].n_vertices

    print(f"  n_verts: {n_verts_level}")
    print(f"  n_faces: {len(faces)}")

    # ===== Template grid =====
    print("\nLoading template grid")
    template_grid = load_template_grid(str(GRID_DIR), HEMI)

    stats = TEMPLATE_SULC_STATS[HEMI]
    template_grid = (template_grid - stats["mean"]) / stats["std"]
    template_grid = template_grid.unsqueeze(0)  # (1, 1, H, W)

    rows = []

    print("\nStarting comparison...\n")

    for sub in SUBJECTS:
        sub_dir = ORIG_ROOT / sub

        old_path = sub_dir / f"{HEMI}.sphere.reg"
        sulc_path = sub_dir / f"{HEMI}.sulc"
        model_path = REG_ROOT / f"{DATASET_NAME}_{sub}_{HEMI}.sphere.reg"

        if not old_path.exists():
            print(f"{sub}: old sphere not found -> {old_path}")
            continue

        if not sulc_path.exists():
            print(f"{sub}: sulc not found -> {sulc_path}")
            continue

        if not model_path.exists():
            print(f"{sub}: model sphere not found -> {model_path}")
            continue

        print(f"Processing {sub}...")

        # ===== Load sulc =====
        sulc_full = nib.freesurfer.read_morph_data(str(sulc_path))
        sulc = sulc_full[:n_verts_level].astype(np.float32).copy()

        sulc_t = torch.from_numpy(sulc).float()
        sulc_t = sulc_t.unsqueeze(0).unsqueeze(0)  # (1, 1, N)
        sulc_t = normalize_sulc(sulc_t)

        # ===== Load old sphere =====
        old_verts_full, _ = nib.freesurfer.read_geometry(str(old_path))
        old_verts = old_verts_full[:n_verts_level].astype(np.float32).copy()

        old_t = torch.from_numpy(old_verts.T).float().unsqueeze(0)  # (1, 3, N)
        old_t = F.normalize(old_t, dim=1) * SPHERE_RADIUS

        # ===== Load new/model sphere =====
        model_verts, _ = nib.freesurfer.read_geometry(str(model_path))
        model_verts = model_verts[:n_verts_level].astype(np.float32).copy()

        model_t = torch.from_numpy(model_verts.T).float().unsqueeze(0)  # (1, 3, N)
        model_t = F.normalize(model_t, dim=1) * SPHERE_RADIUS

        # ===== Sample template sulc =====
        sulc_old = sample_grid_at_vertices(template_grid, old_t)
        sulc_new = sample_grid_at_vertices(template_grid, model_t)

        # ===== MSE =====
        mse_old = torch.nn.functional.mse_loss(sulc_t, sulc_old).item()
        mse_new = torch.nn.functional.mse_loss(sulc_t, sulc_new).item()

        # ===== Correlation =====
        corr_old = pearson_corr(sulc_t, sulc_old)
        corr_new = pearson_corr(sulc_t, sulc_new)

        # ===== Folded triangles =====
        folded_old, folded_old_perc = count_folded_triangles(old_t, faces)
        folded_new, folded_new_perc = count_folded_triangles(model_t, faces)

        rows.append({
            "subject": sub,
            "mse_old": mse_old,
            "mse_new": mse_new,
            "corr_old": corr_old,
            "corr_new": corr_new,
            "folded_old": folded_old,
            "folded_new": folded_new,
            "folded_old_%": folded_old_perc,
            "folded_new_%": folded_new_perc,
        })

    if len(rows) == 0:
        raise RuntimeError("No rows computed. Check SUBJECTS, REG_ROOT and paths.")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV to: {OUT_CSV}")

    print("\n=== TABLE ===")
    print(
        f"{'subject':<10} | "
        f"{'mse_old':>10} {'mse_new':>10} | "
        f"{'corr_old':>10} {'corr_new':>10} | "
        f"{'fold_old':>10} {'fold_new':>10}"
    )
    print("-" * 80)

    for r in rows:
        print(
            f"{r['subject']:<10} | "
            f"{r['mse_old']:>10.4f} {r['mse_new']:>10.4f} | "
            f"{r['corr_old']:>10.4f} {r['corr_new']:>10.4f} | "
            f"{r['folded_old']:>10} {r['folded_new']:>10}"
        )

    print("\n=== MEAN RESULTS ===")
    print(f"MSE old:        {np.mean([r['mse_old'] for r in rows]):.6f}")
    print(f"MSE new:        {np.mean([r['mse_new'] for r in rows]):.6f}")
    print(f"Corr old:       {np.mean([r['corr_old'] for r in rows]):.6f}")
    print(f"Corr new:       {np.mean([r['corr_new'] for r in rows]):.6f}")
    print(f"Folded old:     {np.mean([r['folded_old'] for r in rows]):.2f}")
    print(f"Folded new:     {np.mean([r['folded_new'] for r in rows]):.2f}")
    print(f"Folded old %:   {np.mean([r['folded_old_%'] for r in rows]):.6f}")
    print(f"Folded new %:   {np.mean([r['folded_new_%'] for r in rows]):.6f}")


if __name__ == "__main__":
    main()