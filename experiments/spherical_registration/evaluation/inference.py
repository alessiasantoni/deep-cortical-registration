"""
Generates registered spheres from the ico6_full checkpoint for all TEST
subjects in the given splits, and saves them as .sphere.reg files.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader

from experiments.spherical_registration.dataset import SphereRegDataset
from experiments.spherical_registration.network import SphereRegNet  # adjust the path if different

# ============== CONFIG ==============
CHECKPOINT_PATH = Path(
    "/home/s252125/my_project/experiments/sphere_reg2/runs/ico6_full/best_val_total.pt"
)

DATA_ROOT  = Path("/projects/brainnet-data/sphere-reg")
SPLIT_DIR  = Path("/projects/brainnet-data/subject_splits")
BAD_SURFACES = SPLIT_DIR / "bad_surfaces.json"

HEMI  = "lh"
SPLIT = "test"

DATASETS = [
    "ABIDE", "ADHD200", "ADNI3", "AIBL", "Buckner40",
    "Chinese-HCP", "COBRE", "HCP", "ISBI2015", "MCIC", "OASIS3",
]

OUT_DIR = Path(
    "/home/s252125/my_project/experiments/sphere_reg2/runs/ico6_full_test"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRID_PATH = "/home/s252125/my_project/experiments/spherical_registration/fsaverage_grid/lh.white.features.npz"

TEMPLATE_SULC_STATS = {
    "lh": {"mean": -1.1523885e-08, "std": 0.59384775},
    "rh": {"mean": -4.1938106e-09, "std": 0.5924748},
}
# ====================================


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def standardize_template_grid(template_grid, hemi):
    stats = TEMPLATE_SULC_STATS[hemi]
    return (template_grid - stats["mean"]) / stats["std"]


def load_template_grid(grid_path, hemi):
    grid_data = np.load(grid_path)
    return torch.tensor(grid_data["sulc"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def get_template_sphere_init(hemi, sphere_radius, device):
    import brainsynth.resources
    template_resources = brainsynth.resources.Template()
    tmpl = template_resources.load_surface(hemi, "sphere.reg")["vertices"]
    tmpl = tmpl.T.unsqueeze(0).float().to(device)
    return F.normalize(tmpl, dim=1) * sphere_radius


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    cfg = ckpt.get("config", None)
    if cfg is None:
        ICO_IN, ICO_OUT = 3, 6
        N_ITER = [15, 7, 5, 3]
        BASE_CH = ...
        SPHERE_RADIUS = ...
        STEP_SIZE = ...
    else:
        ICO_IN = cfg["ico_in"]
        ICO_OUT = cfg["ico_out"]
        N_ITER = cfg["n_iter"]
        BASE_CH = cfg["base_channels"]
        SPHERE_RADIUS = cfg["sphere_radius"]
        STEP_SIZE = cfg["step_size"]

    print(f"  ico_in={ICO_IN}, ico_out={ICO_OUT}, n_iter={N_ITER}, "
          f"base_channels={BASE_CH}, sphere_radius={SPHERE_RADIUS}, "
          f"step_size={STEP_SIZE}")

    print(f"Loading template grid from {GRID_PATH}")
    template_grid = load_template_grid(GRID_PATH, HEMI).to(DEVICE)
    template_grid = standardize_template_grid(template_grid, HEMI)
    template_sphere_init = get_template_sphere_init(HEMI, SPHERE_RADIUS, DEVICE)

    print(f"Loading TEST dataset for: {DATASETS}")
    dataset = SphereRegDataset(
        root_dir=DATA_ROOT,
        datasets=DATASETS,
        hemi=HEMI,
        split_dir=SPLIT_DIR,
        split=SPLIT,
        bad_surfaces_file=BAD_SURFACES,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"  total test subjects: {len(dataset)}")

    print("Building model ...")
    model = SphereRegNet(
        ico_in=ICO_IN,
        ico_out=ICO_OUT,
        n_iter=N_ITER,
        in_channels=2,
        base_channels=BASE_CH,
        sphere_radius=SPHERE_RADIUS,
        step_size=STEP_SIZE,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  loaded epoch {ckpt.get('epoch', '?')}")

    faces_np = model.topologies[model.ico_out].faces.cpu().numpy().astype(np.int32)

    print("\nGenerating predictions on TEST subjects ...")
    saved = 0
    with torch.no_grad():
        for batch in loader:
            sulc_subj = batch["sulc_subj"].to(DEVICE)
            B = sulc_subj.shape[0]

            sulc_subj_norm = normalize(sulc_subj)
            grid_for_model = template_grid.expand(B, -1, -1, -1)   # FIX: no extra unsqueeze
            sphere_init = template_sphere_init.expand(B, -1, -1)

            r = model(
                sulc_subj=sulc_subj_norm,
                template_grid=grid_for_model,
                sphere_init=sphere_init,
            )

            for i in range(B):
                subject = batch["subject"][i]
                dataset_name = batch["dataset"][i]
                out_path = OUT_DIR / f"{dataset_name}_{subject}_{HEMI}.sphere.reg"
                pred_np = r[i].T.cpu().numpy().astype(np.float32)
                nib.freesurfer.write_geometry(str(out_path), pred_np, faces_np)
                saved += 1
                if saved % 10 == 0:
                    print(f"  saved {saved}/{len(dataset)}")

    print(f"\nDone. {saved} spheres saved to {OUT_DIR}")


if __name__ == "__main__":
    main()