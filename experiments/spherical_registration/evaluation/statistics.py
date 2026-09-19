"""
Comparison old (FreeSurfer) vs new (model): MSE, Pearson correlation,
and folding (number of triangles + area), on ALL subjects of the full
TEST SET, all datasets together.
"""

from pathlib import Path
import csv

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from brainnet.mesh.topology import DeepSurferTopology
from experiments.spherical_registration.dataset import (
    load_template_grid,
    sample_grid_at_vertices,
)


HEMI = "lh"
MAX_ICO_LEVEL = 6   # builds topologies up to the maximum expected level

SPHERE_RADIUS = 100.0

# Folder with the predicted spheres on the TEST SET
REG_ROOT = Path(
    "/home/s252125/my_project/experiments/spherical_registration/evaluation/runs/ico6_full_test"
)

DATA_ROOT = Path("/projects/brainnet-data/sphere-reg")

GRID_DIR = Path(
    "/home/s252125/my_project/experiments/spherical_registration/fsaverage_grid"
)

OUT_CSV = REG_ROOT / "comparison_old_vs_new_full_testset.csv"


TEMPLATE_SULC_STATS = {
    "lh": {"mean": -1.1523885e-08, "std": 0.59384775},
    "rh": {"mean": -4.1938106e-09, "std": 0.5924748},
}


def normalize_sulc(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten()
    b = b.flatten()
    a = a - a.mean()
    b = b - b.mean()
    corr = (a * b).sum() / (
        torch.sqrt((a ** 2).sum()) * torch.sqrt((b ** 2).sum()) + 1e-8
    )
    return corr.item()


def count_folded_with_area(verts: torch.Tensor, faces: torch.Tensor):
    """Computes both % flipped triangles and % flipped area.

    verts: (1, 3, N)
    faces: (F, 3)
    """
    verts = verts.transpose(1, 2)  # (1, N, 3)

    v0 = verts[:, faces[:, 0]]
    v1 = verts[:, faces[:, 1]]
    v2 = verts[:, faces[:, 2]]

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    areas = 0.5 * normals.norm(dim=-1)

    normals_unit = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
    bary = (v0 + v1 + v2) / 3.0
    bary_unit = bary / (bary.norm(dim=-1, keepdim=True) + 1e-8)
    d = (normals_unit * bary_unit).sum(dim=-1)

    folded = d < 0
    n_folded = folded.sum().item()
    perc_folded = 100.0 * n_folded / faces.shape[0]

    area_folded = areas[folded].sum().item()
    area_total = areas.sum().item()
    perc_area_folded = 100.0 * area_folded / (area_total + 1e-8)

    return n_folded, perc_folded, area_folded, perc_area_folded


def detect_ico_level(n_verts: int, topologies) -> int:
    """Auto-detects the ICO level from the sphere's number of vertices,
    so the script works at any resolution without hardcoding."""
    for level in range(len(topologies)):
        if topologies[level].n_vertices == n_verts:
            return level
    available = [t.n_vertices for t in topologies]
    raise ValueError(
        f"No ICO level has {n_verts} vertices. Available: {available}"
    )


def parse_dataset_subject(fname: str, hemi: str):
    """'ABIDE_sub-0001_lh.sphere.reg' -> ('ABIDE', 'sub-0001')."""
    stem = fname
    suffix = f"_{hemi}.sphere.reg"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    dataset, _, subject = stem.partition("_")
    return dataset, subject


def main():
    print(f"REG_ROOT = {REG_ROOT}")
    print(f"OUT_CSV  = {OUT_CSV}")

    if not REG_ROOT.exists():
        raise RuntimeError(f"REG_ROOT does not exist: {REG_ROOT}")

    # Discovers ALL subjects (all datasets) from the spheres present
    pattern = f"*_{HEMI}.sphere.reg"
    pred_files = sorted(REG_ROOT.glob(pattern))

    print(f"\nFound {len(pred_files)} predicted spheres in {REG_ROOT}")
    if len(pred_files) == 0:
        raise RuntimeError(f"No spheres found matching pattern: {pattern}")

    #  Topologies up to the maximum level (for per-subject auto-detect)
    print(f"\nLoading topologies up to ICO={MAX_ICO_LEVEL}")
    topologies = DeepSurferTopology.recursive_subdivision(MAX_ICO_LEVEL)
    for level in range(len(topologies)):
        print(f"  ico={level}: {topologies[level].n_vertices} vertices, "
              f"{topologies[level].faces.shape[0]} faces")

    #  Template grid 
    print("\nLoading template grid")
    template_grid_full = load_template_grid(str(GRID_DIR), HEMI)
    stats = TEMPLATE_SULC_STATS[HEMI]
    template_grid_full = (template_grid_full - stats["mean"]) / stats["std"]
    template_grid_full = template_grid_full.unsqueeze(0)  # (1, 1, H, W)

    rows = []
    skipped = []

    print(f"\nStarting comparison on {len(pred_files)} subjects...\n")

    for i, model_path in enumerate(pred_files):
        dataset, sub = parse_dataset_subject(model_path.name, HEMI)
        sub_dir = DATA_ROOT / dataset / sub

        old_path = sub_dir / "derivatives" / f"{HEMI}.sphere.reg.freesurfer"
        sulc_path = sub_dir / "derivatives" / f"{HEMI}.sulc"

        if not old_path.exists():
            skipped.append((dataset, sub, "no old (FreeSurfer) sphere"))
            continue
        if not sulc_path.exists():
            skipped.append((dataset, sub, "no sulc"))
            continue

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(pred_files)}] {dataset}/{sub}")

        #  Detects resolution from the file already saved by the model
        model_verts_full, _ = nib.freesurfer.read_geometry(str(model_path))
        n_verts_level = model_verts_full.shape[0]
        ico_level = detect_ico_level(n_verts_level, topologies)
        faces = topologies[ico_level].faces.long()

        #  Load sulc 
        sulc_full = nib.freesurfer.read_morph_data(str(sulc_path))
        sulc = sulc_full[:n_verts_level].astype(np.float32).copy()
        sulc_t = torch.from_numpy(sulc).float().unsqueeze(0).unsqueeze(0)  # (1,1,N)
        sulc_t = normalize_sulc(sulc_t)

        #  Load old (FreeSurfer) sphere 
        old_verts_full, _ = nib.freesurfer.read_geometry(str(old_path))
        old_verts = old_verts_full[:n_verts_level].astype(np.float32).copy()
        old_t = torch.from_numpy(old_verts.T).float().unsqueeze(0)  # (1,3,N)
        old_t = F.normalize(old_t, dim=1) * SPHERE_RADIUS

        #  Load model sphere 
        model_verts = model_verts_full[:n_verts_level].astype(np.float32).copy()
        model_t = torch.from_numpy(model_verts.T).float().unsqueeze(0)
        model_t = F.normalize(model_t, dim=1) * SPHERE_RADIUS

        #  Sample template sulc ====
        sulc_old = sample_grid_at_vertices(template_grid_full, old_t)
        sulc_new = sample_grid_at_vertices(template_grid_full, model_t)

        #  MSE 
        mse_old = torch.nn.functional.mse_loss(sulc_t, sulc_old).item()
        mse_new = torch.nn.functional.mse_loss(sulc_t, sulc_new).item()

        #  Correlation
        corr_old = pearson_corr(sulc_t, sulc_old)
        corr_new = pearson_corr(sulc_t, sulc_new)

        #  Folding (count + area)
        folded_old, folded_old_perc, area_folded_old, perc_area_folded_old = \
            count_folded_with_area(old_t, faces)
        folded_new, folded_new_perc, area_folded_new, perc_area_folded_new = \
            count_folded_with_area(model_t, faces)

        rows.append({
            "dataset": dataset,
            "subject": sub,
            "ico_level": ico_level,
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
        raise RuntimeError("No rows computed. Check REG_ROOT and paths.")

    #  Save CSV 
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV to: {OUT_CSV}")

    #  Skipped subjects 
    if skipped:
        print(f"\n=== SKIPPED ({len(skipped)}) ===")
        for dataset, sub, reason in skipped[:20]:
            print(f"  {dataset}/{sub:<15} {reason}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped)-20} more")

    #  Statistics 
    print(f"\n=== STATISTICS ({len(rows)} subjects) ===")
    print(f"{'metric':<18} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    print("-" * 65)
    for key in ["mse_fs", "mse_new", "corr_fs", "corr_new",
                "folded_fs_%", "folded_new_%", "area_folded_fs_%", "area_folded_new_%"]:
        values = [r[key] for r in rows]
        print(f"{key:<18} "
              f"{np.mean(values):>10.4f} "
              f"{np.std(values):>10.4f} "
              f"{np.min(values):>10.4f} "
              f"{np.max(values):>10.4f}")

    #  Best / worst (GLOBAL, not per-dataset)
    rows_sorted = sorted(rows, key=lambda r: r["mse_new"])
    n_show = min(5, len(rows_sorted))

    print(f"\n=== BEST {n_show} overall (lowest mse_new) ===")
    print(f"{'dataset':<12}{'subject':<14} {'mse_new':>10} {'corr_new':>10} {'folded_new':>12}")
    for r in rows_sorted[:n_show]:
        print(f"  {r['dataset']:<10}{r['subject']:<14} "
              f"{r['mse_new']:>10.4f} {r['corr_new']:>10.4f} {r['folded_new']:>12}")

    print(f"\n=== WORST {n_show} overall (highest mse_new) ===")
    print(f"{'dataset':<12}{'subject':<14} {'mse_new':>10} {'corr_new':>10} {'folded_new':>12}")
    for r in rows_sorted[-n_show:]:
        print(f"  {r['dataset']:<10}{r['subject']:<14} "
              f"{r['mse_new']:>10.4f} {r['corr_new']:>10.4f} {r['folded_new']:>12}")


if __name__ == "__main__":
    main()