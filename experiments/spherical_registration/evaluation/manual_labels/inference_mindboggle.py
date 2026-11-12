"""
Same as evaluation/inference.py, for the Mindboggle-101 subjects: they are not in
the splits, and their sulc is read from derivatives/lh.sulc (FreeSurfer format).

  python -m experiments.spherical_registration.evaluation.manual_labels.inference_mindboggle \
      --data_root <Mindboggle101> --id_map <id_map.txt> --run_dir runs/ico6_train

Writes <run_dir>/mindboggle_predictions/<sub-id>_lh.sphere.reg (or --out_dir).
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from experiments.spherical_registration.dataset import load_template_grid
from experiments.spherical_registration.evaluation.inference import build_model_from_checkpoint
from experiments.spherical_registration.evaluation.manual_labels.common import (
    HEMI, add_data_arguments, load_subjects,
)
from experiments.spherical_registration.finetune_train import (
    DEVICE, GRID_DIR, SPHERE_RADIUS,
    get_template_sphere_init, normalize, standardize_template_grid,
)


def main():
    parser = argparse.ArgumentParser(description="Registered spheres of the Mindboggle-101 subjects.")
    add_data_arguments(parser)
    parser.add_argument("--run_dir", default="runs/ico6_train", help="folder containing best_val_total.pt")
    parser.add_argument("--out_dir", default=None, help="default: <run_dir>/mindboggle_predictions")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ckpt_path = Path(args.run_dir) / "best_val_total.pt"
    output_dir = Path(args.out_dir) if args.out_dir else Path(args.run_dir) / "mindboggle_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{ckpt_path} on {DEVICE}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = build_model_from_checkpoint(ckpt)
    faces_np = model.topologies[model.ico_out].faces.cpu().numpy().astype(np.int32)
    n_verts = model.topologies[model.ico_out].n_vertices

    template_grid = load_template_grid(GRID_DIR, HEMI).to(DEVICE)
    template_grid = standardize_template_grid(template_grid, HEMI)
    template_sphere_init = get_template_sphere_init(HEMI, SPHERE_RADIUS, DEVICE)

    subjects = load_subjects(args.id_map)
    print(f"{len(subjects)} subjects")

    saved = 0
    with torch.no_grad():
        for sub in subjects:
            sulc_path = data_root / sub / "derivatives" / f"{HEMI}.sulc"
            if not sulc_path.exists():
                print(f"  {sub}: no {sulc_path.name}")
                continue
            # .astype: the file is big-endian and torch does not take it as it is
            sulc = nib.freesurfer.read_morph_data(str(sulc_path)).astype(np.float32)
            if sulc.shape[0] != n_verts:
                print(f"  {sub}: sulc has {sulc.shape[0]} vertices instead of {n_verts}")
                continue

            sulc_subj = torch.from_numpy(sulc).float().view(1, 1, -1).to(DEVICE)
            r = model(
                sulc_subj=normalize(sulc_subj),
                template_grid=template_grid.unsqueeze(0),
                sphere_init=template_sphere_init,
            )

            out_path = output_dir / f"{sub}_{HEMI}.sphere.reg"
            nib.freesurfer.write_geometry(str(out_path), r[0].T.cpu().numpy().astype(np.float32), faces_np)
            saved += 1
            if saved % 10 == 0:
                print(f"  saved {saved}/{len(subjects)}")

    print(f"{saved} spheres in {output_dir}")


if __name__ == "__main__":
    main()
