"""
FreeSurfer vs JOSA vs our network, for every predicted sphere in a folder:
- MSE and Pearson correlation between the subject sulc and the template sulc
  sampled at the registered positions
- folded triangles, as count, % of triangles and % of area

  python -m experiments.spherical_registration.evaluation.statistics --run_dir runs/ico6_train

Reads <run_dir>/test_predictions (or --pred_dir), writes <run_dir>/statistics/statistics.csv.
"""

from pathlib import Path
import argparse
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
from experiments.spherical_registration.finetune_train import (
    DATA_ROOT, GRID_DIR, HEMI, SPHERE_RADIUS, standardize_template_grid,
)


MAX_ICO_LEVEL = 6

# the other registrations, read from derivatives/ of each subject
REFERENCE_SPHERES = {
    "fs": f"{HEMI}.sphere.reg.freesurfer",
    "josa": f"{HEMI}.sphere.reg.josa",
}
METHODS = list(REFERENCE_SPHERES) + ["new"]


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
    """verts (1, 3, N), faces (F, 3) -> n. folded, % folded, folded area, % folded area.
    Same orientation test as compute_signed_face_areas in loss.py."""
    verts = verts.transpose(1, 2)

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
    """ico level with this number of vertices."""
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


def load_sphere(path, n_verts):
    """(1, 3, N) sphere of radius SPHERE_RADIUS, cut to the first n_verts vertices."""
    verts, _ = nib.freesurfer.read_geometry(str(path))
    verts = torch.from_numpy(verts[:n_verts].astype(np.float32).T).unsqueeze(0)
    return F.normalize(verts, dim=1) * SPHERE_RADIUS


def main():
    parser = argparse.ArgumentParser(description="FreeSurfer vs JOSA vs model: MSE, correlation, folding.")
    parser.add_argument("--run_dir", default="runs/ico6_train",
                        help="run folder; results go in <run_dir>/statistics")
    parser.add_argument("--pred_dir", default=None,
                        help="folder with the predicted spheres (default: <run_dir>/test_predictions)")
    args = parser.parse_args()

    reg_root = Path(args.pred_dir) if args.pred_dir else Path(args.run_dir) / "test_predictions"
    out_csv = Path(args.run_dir) / "statistics" / "statistics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"{reg_root} -> {out_csv}")

    if not reg_root.exists():
        raise RuntimeError(f"Prediction folder does not exist: {reg_root}")

    # the subjects are simply the spheres found in the folder
    pattern = f"*_{HEMI}.sphere.reg"
    pred_files = sorted(reg_root.glob(pattern))

    print(f"{len(pred_files)} spheres")
    if len(pred_files) == 0:
        raise RuntimeError(f"No spheres found matching pattern: {pattern}")

    topologies = DeepSurferTopology.recursive_subdivision(MAX_ICO_LEVEL)
    for level in range(len(topologies)):
        print(f"  ico={level}: {topologies[level].n_vertices} vertices, "
              f"{topologies[level].faces.shape[0]} faces")

    template_grid_full = load_template_grid(GRID_DIR, HEMI)
    template_grid_full = standardize_template_grid(template_grid_full, HEMI)
    template_grid_full = template_grid_full.unsqueeze(0)  # (1, 1, H, W)

    rows = []
    skipped = []


    for i, model_path in enumerate(pred_files):
        dataset, sub = parse_dataset_subject(model_path.name, HEMI)
        deriv_dir = Path(DATA_ROOT) / dataset / sub / "derivatives"

        sphere_paths = {m: deriv_dir / name for m, name in REFERENCE_SPHERES.items()}
        sphere_paths["new"] = model_path
        sulc_path = deriv_dir / f"{HEMI}.sulc"

        missing = [m for m, p in sphere_paths.items() if not p.exists()]
        if missing:
            skipped.append((dataset, sub, f"no {missing[0]} sphere"))
            continue
        if not sulc_path.exists():
            skipped.append((dataset, sub, "no sulc"))
            continue

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(pred_files)}] {dataset}/{sub}")

        # works at any level: the faces are those of the level of the predicted sphere
        n_verts_level = nib.freesurfer.read_geometry(str(model_path))[0].shape[0]
        ico_level = detect_ico_level(n_verts_level, topologies)
        faces = topologies[ico_level].faces.long()

        sulc_full = nib.freesurfer.read_morph_data(str(sulc_path))
        sulc = sulc_full[:n_verts_level].astype(np.float32).copy()
        sulc_t = torch.from_numpy(sulc).float().unsqueeze(0).unsqueeze(0)  # (1,1,N)
        sulc_t = normalize_sulc(sulc_t)

        row = {"dataset": dataset, "subject": sub, "ico_level": ico_level}
        metrics = {}
        for m, path in sphere_paths.items():
            sphere = load_sphere(path, n_verts_level)
            sulc_template = sample_grid_at_vertices(template_grid_full, sphere)
            n_folded, perc_folded, _, perc_area_folded = count_folded_with_area(sphere, faces)
            metrics[m] = {
                "mse": F.mse_loss(sulc_t, sulc_template).item(),
                "corr": pearson_corr(sulc_t, sulc_template),
                "folded": n_folded,
                "folded_%": perc_folded,
                "area_folded_%": perc_area_folded,
            }
        # Column order: mse_fs, mse_josa, mse_new, corr_fs, ..., area_folded_new_%
        for m in METHODS:
            row[f"mse_{m}"] = metrics[m]["mse"]
        for m in METHODS:
            row[f"corr_{m}"] = metrics[m]["corr"]
        for m in METHODS:
            row[f"folded_{m}"] = metrics[m]["folded"]
        for m in METHODS:
            row[f"folded_{m}_%"] = metrics[m]["folded_%"]
        for m in METHODS:
            row[f"area_folded_{m}_%"] = metrics[m]["area_folded_%"]
        rows.append(row)

    if len(rows) == 0:
        raise RuntimeError("No rows computed. Check --run_dir / --pred_dir and the data paths.")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{out_csv}")

    if skipped:
        print(f"\nskipped {len(skipped)}:")
        for dataset, sub, reason in skipped[:20]:
            print(f"  {dataset}/{sub:<15} {reason}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped)-20} more")

    print(f"\n{len(rows)} subjects")
    print(f"{'metric':<18} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    print("-" * 65)
    keys = ([f"mse_{m}" for m in METHODS] + [f"corr_{m}" for m in METHODS]
            + [f"folded_{m}_%" for m in METHODS] + [f"area_folded_{m}_%" for m in METHODS])
    for key in keys:
        values = [r[key] for r in rows]
        print(f"{key:<18} "
              f"{np.mean(values):>10.4f} "
              f"{np.std(values):>10.4f} "
              f"{np.min(values):>10.4f} "
              f"{np.max(values):>10.4f}")

    # best and worst over all datasets together
    rows_sorted = sorted(rows, key=lambda r: r["mse_new"])
    n_show = min(5, len(rows_sorted))

    print(f"\nlowest mse_new:")
    print(f"{'dataset':<12}{'subject':<14} {'mse_new':>10} {'corr_new':>10} {'folded_new':>12}")
    for r in rows_sorted[:n_show]:
        print(f"  {r['dataset']:<10}{r['subject']:<14} "
              f"{r['mse_new']:>10.4f} {r['corr_new']:>10.4f} {r['folded_new']:>12}")

    print(f"\nhighest mse_new:")
    print(f"{'dataset':<12}{'subject':<14} {'mse_new':>10} {'corr_new':>10} {'folded_new':>12}")
    for r in rows_sorted[-n_show:]:
        print(f"  {r['dataset']:<10}{r['subject']:<14} "
              f"{r['mse_new']:>10.4f} {r['corr_new']:>10.4f} {r['folded_new']:>12}")


if __name__ == "__main__":
    main()
