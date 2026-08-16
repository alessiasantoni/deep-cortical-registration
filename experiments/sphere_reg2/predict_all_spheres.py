"""
In this script I save all the sphere.reg predicted by the model, 
using a saved checkpoint from one old training run.
It can be used to test the network on an unseen dataset.

Apply "statistics_all.py" after to see the statistics.
"""

from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.sphere_reg2.dataset import SphereRegDataset, load_template_grid
from experiments.sphere_reg2.network_rotation import SphereRegNet

# CONFIG 

CHECKPOINT_PATH = Path("experiments/sphere_reg2/runs/20260609_181249_finetune_ico6/best_val_total.pt")
OUTPUT_DIR      = Path("experiments/sphere_reg2/runs/20260609_181249_finetune_ico6/spheres_all/")
DATA_ROOT  = "/projects/brainnet-data/sphere-reg"
GRID_DIR   = "experiments/sphere_reg2/fsaverage_grid"
DATASETS   = ["COBRE"]
SPLIT_DIR = "/projects/brainnet-data/subject_splits"
SPLIT      = "test"
HEMI       = "lh"


# Devono matchare l'architettura del checkpoint
ICO_IN     = 3
ICO_OUT    = 6
N_ITER     = [15, 7, 5, 3]
BASE_CH    = 32
STEP_SIZE  = 0.01  #0.01
SPHERE_RADIUS = 100.0


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Helpers (copiati dal training script)

def normalize(x, dims=None):
    if dims is None:
        dims = tuple(range(1, x.ndim))
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True) + 1e-8
    return (x - mean) / std


TEMPLATE_SULC_STATS = {
    "lh": {"mean": -1.1523885e-08, "std": 0.59384775},
    "rh": {"mean": -4.1938106e-09, "std":  0.5924748},
}

def standardize_template_grid(template_grid, hemi):
    stats = TEMPLATE_SULC_STATS[hemi]
    return (template_grid - stats["mean"]) / stats["std"]


def get_template_sphere_init(hemi, sphere_radius, device):
    import brainsynth.resources
    template_resources = brainsynth.resources.Template()
    tmpl = template_resources.load_surface(hemi, "sphere.reg")["vertices"]
    tmpl = tmpl.T.unsqueeze(0).float().to(device)
    return F.normalize(tmpl, dim=1) * sphere_radius


# Setup

print(f"Device: {DEVICE}")
print(f"Loading template grid ...")
template_grid = load_template_grid(GRID_DIR, HEMI).to(DEVICE)
template_grid = standardize_template_grid(template_grid, HEMI)

template_sphere_init = get_template_sphere_init(HEMI, SPHERE_RADIUS, DEVICE)

print(f"Loading dataset ...")
#dataset = SphereRegDataset(root_dir=DATA_ROOT, datasets=DATASETS, hemi=HEMI)

dataset = SphereRegDataset(
    root_dir=DATA_ROOT,
    datasets=DATASETS,
    hemi=HEMI,
    split_dir=SPLIT_DIR,
    split=SPLIT,
)

loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
print(f"  total subjects: {len(dataset)}")

print(f"Building model ...")
model = SphereRegNet(
    ico_in=ICO_IN, ico_out=ICO_OUT, n_iter=N_ITER,
    in_channels=2, base_channels=BASE_CH,
    sphere_radius=SPHERE_RADIUS, step_size=STEP_SIZE,
).to(DEVICE)

print(f"Loading checkpoint: {CHECKPOINT_PATH}")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"  loaded epoch {ckpt['epoch']}")

faces_np = model.topologies[model.ico_out].faces.cpu().numpy().astype(np.int32)


# Inferenza su tutti i soggetti

print(f"\nGenerating predictions ...")
saved = 0
with torch.no_grad():
    for batch in loader:
        sulc_subj = batch["sulc_subj"].to(DEVICE)
        B = sulc_subj.shape[0]

        sulc_subj_norm = normalize(sulc_subj)
        grid_for_model = template_grid.unsqueeze(0).expand(B, -1, -1, -1)
        sphere_init = template_sphere_init.expand(B, -1, -1)

        r = model(
            sulc_subj=sulc_subj_norm,
            template_grid=grid_for_model,
            sphere_init=sphere_init,
        )

        for i in range(B):
            subject = batch["subject"][i]
            dataset_name = batch["dataset"][i]
            out_path = OUTPUT_DIR / f"{dataset_name}_{subject}_{HEMI}.sphere.reg"
            pred_np = r[i].T.cpu().numpy().astype(np.float32)
            nib.freesurfer.write_geometry(str(out_path), pred_np, faces_np)
            saved += 1
            if saved % 10 == 0:
                print(f"  saved {saved}/{len(dataset)}")

print(f"\nDone. {saved} spheres saved to {OUTPUT_DIR}")