"""
Runs a trained checkpoint on every subject of a split and saves the
registered spheres as FreeSurfer .sphere.reg files.

The model is rebuilt from the checkpoint itself: the finest level is read
from the weights, the ico3 depth from "val_reference_iter", and the depths of
the finer levels from FINE_N_ITER in finetune_train.py.

Usage (from the repository root):
  python -m experiments.spherical_registration.evaluation.inference --run_dir runs/ico6_train
Output: <run_dir>/<split>_predictions/<dataset>_<subject>_<hemi>.sphere.reg

python -m experiments.spherical_registration.evaluation.inference --run_dir runs/ico6_train

"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.spherical_registration.dataset import SphereRegDataset, load_template_grid
from experiments.spherical_registration.network import SphereRegNet
from experiments.spherical_registration.finetune_train import (
    BAD_SURFACES, BASE_CH, DATA_ROOT, DEVICE, FINE_N_ITER, GRID_DIR, HEMI, ICO_IN,
    SPHERE_RADIUS, SPLIT_DIR, STEP_SIZE,
    get_template_sphere_init, normalize, standardize_template_grid,
)

# Includes COBRE and MCIC, which have only a test split (unseen during training)
DATASETS = [
    "ABIDE", "ADHD200", "ADNI3", "AIBL", "Buckner40",
    "Chinese-HCP", "COBRE", "HCP", "ISBI2015", "MCIC", "OASIS3",
]


def build_model_from_checkpoint(ckpt):
    state_dict = ckpt["model_state_dict"]
    levels = sorted({int(k.split(".")[1]) for k in state_dict if k.startswith("unets.")})
    ico_out = levels[-1]
    n_iter = [ckpt["val_reference_iter"]] + [FINE_N_ITER[l] for l in range(ICO_IN + 1, ico_out + 1)]

    model = SphereRegNet(
        ico_in=ICO_IN, ico_out=ico_out, n_iter=n_iter, in_channels=2,
        base_channels=BASE_CH, sphere_radius=SPHERE_RADIUS, step_size=STEP_SIZE,
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  ico{ICO_IN}->ico{ico_out}, n_iter={n_iter}, epoch {ckpt.get('epoch', '?')}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Save registered spheres for one split.")
    parser.add_argument("--run_dir", default="runs/ico6_train",
                        help="folder containing best_val_total.pt")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "best_val_total.pt"
    out_dir = run_dir / f"{args.split}_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = build_model_from_checkpoint(ckpt)
    faces_np = model.topologies[model.ico_out].faces.cpu().numpy().astype(np.int32)

    template_grid = load_template_grid(GRID_DIR, HEMI).to(DEVICE)
    template_grid = standardize_template_grid(template_grid, HEMI)
    template_sphere_init = get_template_sphere_init(HEMI, SPHERE_RADIUS, DEVICE)

    dataset = SphereRegDataset(root_dir=DATA_ROOT, datasets=DATASETS, hemi=HEMI,
                               split_dir=SPLIT_DIR, split=args.split, bad_surfaces_file=BAD_SURFACES)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"{args.split} subjects: {len(dataset)}")

    saved = 0
    with torch.no_grad():
        for batch in loader:
            sulc_subj = batch["sulc_subj"].to(DEVICE)
            B = sulc_subj.shape[0]

            r = model(
                sulc_subj=normalize(sulc_subj),
                template_grid=template_grid.unsqueeze(0).expand(B, -1, -1, -1),
                sphere_init=template_sphere_init.expand(B, -1, -1),
            )

            for i in range(B):
                out_path = out_dir / f"{batch['dataset'][i]}_{batch['subject'][i]}_{HEMI}.sphere.reg"
                nib.freesurfer.write_geometry(str(out_path), r[i].T.cpu().numpy().astype(np.float32), faces_np)
                saved += 1
                if saved % 50 == 0:
                    print(f"  saved {saved}/{len(dataset)}")

    print(f"Done. {saved} spheres saved to {out_dir}")


if __name__ == "__main__":
    main()
