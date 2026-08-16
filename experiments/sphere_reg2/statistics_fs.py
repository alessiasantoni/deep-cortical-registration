
from pathlib import Path
import csv

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from brainnet.mesh.topology import DeepSurferTopology
from experiments.sphere_reg2.dataset import (
    load_template_grid,
    sample_grid_at_vertices,
)

# ============== CONFIG ==============
HEMI = "lh"
DATASET_NAME   = "OASIS3"
EPOCH = 25
ICO_LEVEL = 3

SPHERE_RADIUS = 100.0

RUN_DIR = Path("/home/s252125/my_project/experiments/sphere_reg2/runs/20260609_181249_finetune_ico6")  # directory del training run da cui prendere le spheres
ORIG_ROOT = Path(f"/projects/brainnet-data/sphere-reg/{DATASET_NAME}")

#REG_ROOT = RUN_DIR / "spheres" / f"epoch_{EPOCH:03d}"
REG_ROOT = RUN_DIR / "spheres_all"


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

def count_folded_with_area(verts: torch.Tensor, faces: torch.Tensor):
    """Calcola sia % triangoli flippati che % area flippata.
    
    verts: (1, 3, N)
    faces: (F, 3)
    
    Returns:
        n_folded:        numero di triangoli flippati
        perc_folded:     % triangoli flippati (in numero)
        area_folded:     area totale dei triangoli flippati
        perc_area_folded: % di area flippata sul totale
    """
    verts = verts.transpose(1, 2)  # (1, N, 3)
    
    v0 = verts[:, faces[:, 0]]
    v1 = verts[:, faces[:, 1]]
    v2 = verts[:, faces[:, 2]]
    
    # Normali geometriche (modulo = 2 * area senza segno)
    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    areas = 0.5 * normals.norm(dim=-1)  # (1, F)  -- area assoluta di ogni triangolo
    
    normals_unit = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
    bary = (v0 + v1 + v2) / 3.0
    bary_unit = bary / (bary.norm(dim=-1, keepdim=True) + 1e-8)
    d = (normals_unit * bary_unit).sum(dim=-1)
    
    folded = d < 0  # (1, F)
    
    # Conteggio triangoli
    n_folded = folded.sum().item()
    perc_folded = 100.0 * n_folded / faces.shape[0]
    
    # Conteggio in area
    area_folded = areas[folded].sum().item()
    area_total = areas.sum().item()
    perc_area_folded = 100.0 * area_folded / (area_total + 1e-8)
    
    return n_folded, perc_folded, area_folded, perc_area_folded

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

    # ===== Discover subjects from saved spheres =====
    # File pattern: <DATASET>_<subject>_<HEMI>.sphere.reg
    # es: OASIS3_sub-0026_lh.sphere.reg
    pattern = f"{DATASET_NAME}_*_{HEMI}.sphere.reg"
    pred_files = sorted(REG_ROOT.glob(pattern))

    prefix = f"{DATASET_NAME}_"
    suffix = f"_{HEMI}.sphere.reg"
    subjects = [
        p.name.removeprefix(prefix).removesuffix(suffix)
        for p in pred_files
    ]

    print(f"\nFound {len(subjects)} predicted spheres in {REG_ROOT}")
    if len(subjects) == 0:
        raise RuntimeError(f"No spheres found matching pattern: {pattern}")

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
    skipped = []

    print(f"\nStarting comparison on {len(subjects)} subjects...\n")

    for i, sub in enumerate(subjects):
        sub_dir = ORIG_ROOT / sub

        old_path = sub_dir / "derivatives" / f"{HEMI}.sphere.reg.freesurfer"
        sulc_path = sub_dir / "derivatives" / f"{HEMI}.sulc"
        model_path = REG_ROOT / f"{DATASET_NAME}_{sub}_{HEMI}.sphere.reg"

        if not old_path.exists():
            skipped.append((sub, "no old sphere"))
            continue

        if not sulc_path.exists():
            skipped.append((sub, "no sulc"))
            continue

        if not model_path.exists():
            skipped.append((sub, "no model sphere"))
            continue

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(subjects)}] {sub}")

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

        folded_old, folded_old_perc, area_folded_old, perc_area_folded_old = \
            count_folded_with_area(old_t, faces)
        folded_new, folded_new_perc, area_folded_new, perc_area_folded_new = \
            count_folded_with_area(model_t, faces)

        rows.append({
            "subject": sub,
            "mse_fs": mse_old,
            "mse_new": mse_new,
            "corr_fs": corr_old,
            "corr_new": corr_new,
            "folded_fs": folded_old,
            "folded_new": folded_new,
            "folded_fs_%": folded_old_perc,
            "folded_new_%": folded_new_perc,
            "area_folded_fs_%": perc_area_folded_old,
            "area_folded_new_%": perc_area_folded_new,
        })

    if len(rows) == 0:
        raise RuntimeError("No rows computed. Check subjects, REG_ROOT and paths.")

    # ===== Save CSV =====
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV to: {OUT_CSV}")

    # ===== Skipped subjects =====
    if skipped:
        print(f"\n=== SKIPPED ({len(skipped)}) ===")
        for sub, reason in skipped:
            print(f"  {sub:<15} {reason}")

    # ===== Statistics =====
    print(f"\n=== STATISTICS ({len(rows)} subjects) ===")
    print(f"{'metric':<15} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    print("-" * 60)
    for key in ["mse_fs", "mse_new", "corr_fs", "corr_new",
                "folded_fs_%", "folded_new_%", "area_folded_fs_%", "area_folded_new_%"]:
        values = [r[key] for r in rows]
        print(f"{key:<15} "
              f"{np.mean(values):>10.4f} "
              f"{np.std(values):>10.4f} "
              f"{np.min(values):>10.4f} "
              f"{np.max(values):>10.4f}")

    # ===== Best / worst =====
    rows_sorted = sorted(rows, key=lambda r: r["mse_new"])

    n_show = min(5, len(rows_sorted))

    print(f"\n=== BEST {n_show} (lowest mse_new) ===")
    print(f"{'subject':<15} {'mse_new':>10} {'corr_new':>10} {'folded_new':>12}")
    for r in rows_sorted[:n_show]:
        print(f"  {r['subject']:<13} "
              f"{r['mse_new']:>10.4f} "
              f"{r['corr_new']:>10.4f} "
              f"{r['folded_new']:>12}")

    print(f"\n=== WORST {n_show} (highest mse_new) ===")
    print(f"{'subject':<15} {'mse_new':>10} {'corr_new':>10} {'folded_new':>12}")
    for r in rows_sorted[-n_show:]:
        print(f"  {r['subject']:<13} "
              f"{r['mse_new']:>10.4f} "
              f"{r['corr_new']:>10.4f} "
              f"{r['folded_new']:>12}")


if __name__ == "__main__":
    main()