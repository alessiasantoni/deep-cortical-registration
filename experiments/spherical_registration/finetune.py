"""
Fine tuning code.

It is used to load the previous resolution level weights and freeze them to fine tune
the upper resolution levels.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import csv

from experiments.sphere_reg2.dataset import (
    SphereRegDataset,
    load_template_grid,
    sample_grid_at_vertices,
)
from experiments.sphere_reg2.network_rotation import SphereRegNet
from experiments.sphere_reg2.loss_total import SulcalAlignmentLoss


# CONFIG

SPLIT_DIR = "/projects/brainnet-data/subject_splits"
BAD_SURFACES = "/projects/brainnet-data/subject_splits/bad_surfaces.json"

DATA_ROOT  = "/projects/brainnet-data/sphere-reg"
GRID_DIR   = "experiments/sphere_reg2/fsaverage_grid"
DATASETS = ["ABIDE", "ADHD200", "ADNI3", "AIBL", "Buckner40",
            "Chinese-HCP", "HCP", "OASIS3", "ISBI2015"]
# COBRE and MCIC datasets excluded from the training
HEMI       = "lh"

# Checkpoints from previous levels
ICO3_CHECKPOINT = Path(
    "experiments/sphere_reg2/runs/ico3_full/best_val_total.pt"
)
ICO4_CHECKPOINT = Path(
    "experiments/sphere_reg2/runs/ico4_full/best_val_total.pt"
)
ICO5_CHECKPOINT = Path(
    "experiments/sphere_reg2/runs/ico5_full/best_val_total.pt"
)


#  Risoluzion levels 
ICO_IN     = 3
ICO_OUT    = 6

#  Iterazions 
N_ITER     = [15, 7, 5, 3]
BASE_CH    = 32
STEP_SIZE  = 0.01     
SPHERE_RADIUS = 100.0

#  Training 
N_EPOCHS   = 25
BATCH_SIZE = 1
#LR         = 1e-5      # 1e-5 for the first finetuning (15362 verts)
LR         = 1e-4       # 1e-4 for the second and third finetunings (61442 and 245762 verts)
WEIGHT_DECAY = 1e-4 

#  Loss weights 
W_SULC     = 1.0         
#W_METRIC   = 0.25     # 15362 verts
#W_METRIC   = 0.1      # 61442 verts
W_METRIC   = 0.05      # 245762 verts
W_AREA     = 0.05     
W_COUNT    = 0.00001   # 15362 verts
W_COUNT    = 0.000001  # 61442 and 245762 verts

COUNT_TEMPERATURE = 1.0
FOLDED_REDUCTION = "sum"  


#  Save 
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = f"experiments/sphere_reg2/runs/{timestamp}_ico3_full"
SAVE_EVERY = 5
N_SAVE     = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Early stopping
EARLY_STOPPING_PATIENCE = 3      # epochs without improvement before stopping
EARLY_STOPPING_MIN_DELTA = 1e-4  # minimal improvement to reset patience

# Helpers

def normalize(x: torch.Tensor, dims: tuple = None) -> torch.Tensor:
    if dims is None:
        dims = tuple(range(1, x.ndim))
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True) + 1e-8
    return (x - mean) / std


TEMPLATE_SULC_STATS = {
    "lh": {"mean": -1.1523885e-08, "std": 0.59384775},
    "rh": {"mean": -4.1938106e-09, "std":  0.5924748},
}


def standardize_template_grid(template_grid: torch.Tensor, hemi: str) -> torch.Tensor:
    stats = TEMPLATE_SULC_STATS[hemi]
    return (template_grid - stats["mean"]) / stats["std"]


def get_template_sphere_init(hemi: str, sphere_radius: float, device) -> torch.Tensor:
    import brainsynth.resources
    template_resources = brainsynth.resources.Template()
    template_sphere = template_resources.load_surface(hemi, "sphere.reg")["vertices"]
    template_sphere = template_sphere.T.unsqueeze(0).float().to(device)
    return F.normalize(template_sphere, dim=1) * sphere_radius


# Checkpoint loading + Freeze

def load_lower_weights(model: SphereRegNet, ckpt_path: Path):
    """Uploads the weights of the sub-network corresponding to ico=3 from the checkpoint into the ico=4 model.
    """
    print(f"\nLoading ico=3 checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        print(f"  Checkpoint metadata: epoch={ckpt.get('epoch', '?')}")
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"\n  Transfer summary:")
    print(f"    Loaded keys:     {len(state_dict)}")
    print(f"    Missing keys:    {len(missing)} (will train from scratch)")
    print(f"    Unexpected keys: {len(unexpected)}")

    if unexpected:
        print(f"\n  WARNING: unexpected keys in checkpoint:")
        for k in unexpected[:5]:
            print(f"    - {k}")

    return model

def freeze_lower_subnets(model, freeze_orders=("3", "4")):
    """Freeze all parameters of the sub-network
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(f"unets.{o}" in name or f"deforms.{o}" in name for o in freeze_orders):
            param.requires_grad = False
            frozen_count += param.numel()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    #print(f"\n  Freezing sub-networks:")
    print(f"\n  Freezing sub-networks {freeze_orders}:")
    print(f"    Frozen params:    {frozen_count:,}")
    print(f"    Trainable params: {trainable:,}")
    print(f"    Total params:     {total:,}")
    print(f"    Frozen / Total:   {100*frozen_count/total:.1f}%")


# Sphere saving

def save_predicted_spheres(model, loader, template_grid, template_sphere_init,
                            epoch, save_dir, n_save=N_SAVE):
    out_dir = save_dir / "spheres" / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    faces_np = model.topologies[model.ico_out].faces.cpu().numpy().astype(np.int32)

    model.eval()
    saved = 0
    with torch.no_grad():
        for batch in loader:
            if saved >= n_save:
                break
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
            n_out = r.shape[-1]

            for i in range(min(B, n_save - saved)):
                subject = batch["subject"][i]
                dataset_name = batch["dataset"][i]
                stem = out_dir / f"{dataset_name}_{subject}_{HEMI}"
                pred_np = r[i].T.cpu().numpy().astype(np.float32)
                init_np = sphere_init[i, :, :n_out].T.cpu().numpy().astype(np.float32)
                nib.freesurfer.write_geometry(str(stem) + ".sphere.reg", pred_np, faces_np)
                nib.freesurfer.write_geometry(str(stem) + ".sphere.input", init_np, faces_np)
                saved += 1

    print(f"  Saved {saved} sphere(s) -> {out_dir}")


# Diagnostic

def diagnostic_forward(model, loader, template_grid, template_sphere_init):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            sulc_subj = batch["sulc_subj"].to(DEVICE)
            B = sulc_subj.shape[0]

            sulc_subj_norm = normalize(sulc_subj)
            grid_for_model = template_grid.unsqueeze(0).expand(B, -1, -1, -1)
            sphere_init = template_sphere_init.expand(B, -1, -1)

            r, stats = model(
                sulc_subj=sulc_subj_norm,
                template_grid=grid_for_model,
                sphere_init=sphere_init,
                return_intermediate=True,
            )

            print(f"\n=== Diagnostic forward (subject: {batch['subject'][0]}) ===")
            print(f"{'order':>6} {'iter':>5} {'verts':>8} {'|dr|_mean':>10} "
                  f"{'|dr|_max':>10} {'dr_std':>10}")
            for s in stats:
                print(f"{s['order']:>6} {s['iter']:>5} {s['n_verts']:>8} "
                      f"{s['dr_mean']:>10.4f} {s['dr_max']:>10.4f} {s['dr_std']:>10.4f}")
            print()
            break
    if was_training:
        model.train()


# Run epoch

def run_epoch(model, loader, criterion, template_grid, template_sphere_init, optimizer):
    is_train = optimizer is not None
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    sums = {}
    batch_count = 0
    n_valid_batches = 0

    with ctx:
        for batch in loader:
            sulc_subj = batch["sulc_subj"].to(DEVICE)
            wm_verts = batch["wm_verts"].to(DEVICE)
            B = sulc_subj.shape[0]

            sulc_subj_norm = normalize(sulc_subj)
            grid_for_model = template_grid.unsqueeze(0).expand(B, -1, -1, -1)
            sphere_init = template_sphere_init.expand(B, -1, -1)

            r = model(
                sulc_subj=sulc_subj_norm,
                template_grid=grid_for_model,
                sphere_init=sphere_init,
            )
            n_out = r.shape[-1]

            loss_dict = criterion(
                pred_verts=r,
                sulc_subj=sulc_subj_norm[:, :, :n_out],
                template_grid=grid_for_model,
                topology=model.topologies[model.ico_out],
                ref_verts_metric=wm_verts[:, :, :n_out],
                ref_verts_area=sphere_init[:, :, :n_out],
            )

            if torch.isnan(loss_dict["total"]) or torch.isnan(r).any():
                print(f"  [batch {batch_count}] NaN, subject: {batch['subject'][0]} - SKIPPING")
                if is_train:
                    optimizer.zero_grad()
                batch_count += 1
                continue

            if is_train:
                optimizer.zero_grad()
                loss_dict["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=GRAD_CLIP
                )

                has_nan_grad = any(
                    p.grad is not None and torch.isnan(p.grad).any()
                    for p in model.parameters() if p.requires_grad
                )
                if has_nan_grad:
                    print(f"  [batch {batch_count}] NaN gradients - SKIPPING")
                    optimizer.zero_grad()
                    batch_count += 1
                    continue

                if batch_count < 3:
                    msg = (
                        f"  [batch {batch_count}] "
                        f"sulc={loss_dict.get('sulc', torch.tensor(0.0)).item():.4f}  "
                        f"metric={loss_dict.get('metric', torch.tensor(0.0)).item():.4f}  "
                    )

                    if "area" in loss_dict:
                        msg += f"area={loss_dict['area'].item():.4e}  "

                    if "count" in loss_dict:
                        msg += f"count={loss_dict['count'].item():.4e}"

                    print(msg)

                optimizer.step()

            for k, v in loss_dict.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            n_valid_batches += 1
            batch_count += 1

    n_valid = max(n_valid_batches, 1)
    return {k: v / n_valid for k, v in sums.items()}

# Validation evaluation (subject-by-subject)

def evaluate_val_subjects(model, loader, criterion, template_grid, template_sphere_init):
    """
    Computes validation loss subject-by-subject.

    Assumes BATCH_SIZE = 1 for clean subject-level statistics.
    Returns:
        summary: dict with mean/std/best/worst
        rows: list of dicts, one per subject
    """
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in loader:
            sulc_subj = batch["sulc_subj"].to(DEVICE)
            wm_verts = batch["wm_verts"].to(DEVICE)
            B = sulc_subj.shape[0]

            if B != 1:
                raise ValueError(
                    "evaluate_val_subjects expects BATCH_SIZE=1. "
                    "Use a separate val_stats_loader with batch_size=1."
                )

            subject = batch["subject"][0]
            dataset_name = batch["dataset"][0]

            sulc_subj_norm = normalize(sulc_subj)
            grid_for_model = template_grid.unsqueeze(0).expand(B, -1, -1, -1)
            sphere_init = template_sphere_init.expand(B, -1, -1)

            r = model(
                sulc_subj=sulc_subj_norm,
                template_grid=grid_for_model,
                sphere_init=sphere_init,
            )

            n_out = r.shape[-1]

            loss_dict = criterion(
                pred_verts=r,
                sulc_subj=sulc_subj_norm[:, :, :n_out],
                template_grid=grid_for_model,
                topology=model.topologies[model.ico_out],
                ref_verts_metric=wm_verts[:, :, :n_out],
                ref_verts_area=sphere_init[:, :, :n_out],
            )

            rows.append({
                "dataset": dataset_name,
                "subject": subject,
                "total": loss_dict["total"].item(),
                "sulc": loss_dict.get("sulc", torch.tensor(0.0)).item(),
                "metric": loss_dict.get("metric", torch.tensor(0.0)).item(),
                "area": loss_dict.get("area", torch.tensor(0.0)).item(),
                "count": loss_dict.get("count", torch.tensor(0.0)).item(),
            })

    total_values = np.array([r["total"] for r in rows], dtype=np.float64)

    best_row = min(rows, key=lambda x: x["total"])
    worst_row = max(rows, key=lambda x: x["total"])

    summary = {
        "val_total_mean": float(total_values.mean()),
        "val_total_std": float(total_values.std(ddof=1)) if len(total_values) > 1 else 0.0,
        "best_dataset": best_row["dataset"],
        "best_subject": best_row["subject"],
        "best_total": best_row["total"],
        "worst_dataset": worst_row["dataset"],
        "worst_subject": worst_row["subject"],
        "worst_total": worst_row["total"],
    }

    return summary, rows


# Folding counter

def count_folded_full(model, loader, template_grid, template_sphere_init, faces):
    """Counts the number of folded faces across the entire validation set."""
    model.eval()
    total_folded = 0
    total_faces = 0
    with torch.no_grad():
        for batch in loader:
            sulc_subj = batch["sulc_subj"].to(DEVICE)
            B = sulc_subj.shape[0]
            sulc_subj_norm = normalize(sulc_subj)
            grid_for_model = template_grid.unsqueeze(0).expand(B, -1, -1, -1)
            sphere_init = template_sphere_init.expand(B, -1, -1)
            r = model(sulc_subj=sulc_subj_norm, template_grid=grid_for_model,
                      sphere_init=sphere_init)

            verts = r.transpose(1, 2)
            v0 = verts[:, faces[:, 0]]
            v1 = verts[:, faces[:, 1]]
            v2 = verts[:, faces[:, 2]]
            normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
            normals_unit = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
            bary = (v0 + v1 + v2) / 3.0
            bary_unit = bary / (bary.norm(dim=-1, keepdim=True) + 1e-8)
            d = (normals_unit * bary_unit).sum(dim=-1)
            folded = (d < 0).sum().item()
            total_folded += folded
            total_faces += B * faces.shape[0]

    return 100.0 * total_folded / total_faces


# Main

def main():
    print(f"Device: {DEVICE}")
    print(f"Save dir: {SAVE_DIR}")

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    
    print("\nLoading template grid ...")
    template_grid = load_template_grid(GRID_DIR, HEMI).to(DEVICE)
    template_grid = standardize_template_grid(template_grid, HEMI)

    template_sphere_init = get_template_sphere_init(HEMI, SPHERE_RADIUS, DEVICE)

   

    train_ds = SphereRegDataset(root_dir=DATA_ROOT, datasets=DATASETS, hemi=HEMI,
                                split_dir=SPLIT_DIR, split="train",
                                bad_surfaces_file=BAD_SURFACES)
    val_ds   = SphereRegDataset(root_dir=DATA_ROOT, datasets=DATASETS, hemi=HEMI,
                                split_dir=SPLIT_DIR, split="validation",
                                bad_surfaces_file=BAD_SURFACES)


    print(f"\n{'='*50}")
    print(f"TOTAL Train: {len(train_ds)} subjects across {len(DATASETS)} datasets")
    print(f"TOTAL Val:   {len(val_ds)} subjects")
    print(f"{'='*50}\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    val_stats_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print(f"\nBuilding model: ICO_IN={ICO_IN}, ICO_OUT={ICO_OUT}, N_ITER={N_ITER}")
    model = SphereRegNet(
        ico_in=ICO_IN,
        ico_out=ICO_OUT,
        n_iter=N_ITER,
        in_channels=2,
        base_channels=BASE_CH,
        sphere_radius=SPHERE_RADIUS,
        step_size=STEP_SIZE,
    ).to(DEVICE)

    print(f"  Sub-nets present: {list(model.unets.keys())}")
 
    # FREEZING of weights from previous resolution levels
    
    # load the desired checkpoint (3, 4, 5)
    load_lower_weights(model, ICO5_CHECKPOINT)     

    # freeze the lower subnets (3, 4, 5) to fine tune the upper levels
    #freeze_lower_subnets(model, freeze_orders=["3"])          # to freeze the first resolution level
    #freeze_lower_subnets(model, freeze_orders=[3, 4])         # to freeze the first and the second resolution levels
    freeze_lower_subnets(model, freeze_orders=[3, 4, 5])      # to freeze the first, second and the third resolution levels
 
    print("\n=== Parametri trainable per sub-rete ===")
    for order in ["3", "4", "5"]:
        n_train = sum(p.numel() for name, p in model.named_parameters()
                      if f".{order}" in name and p.requires_grad)
        n_total = sum(p.numel() for name, p in model.named_parameters()
                      if f".{order}" in name)
        print(f"  ico={order}: {n_train:,} trainable / {n_total:,} total")

    #  Loss & optimizer
    criterion = SulcalAlignmentLoss(
        w_sulc=W_SULC,
        w_metric=W_METRIC,
        w_area=W_AREA,
        folded_reduction=FOLDED_REDUCTION,
        w_count=W_COUNT,
        #w_smooth=W_SMOOTH,                       
        count_temperature=COUNT_TEMPERATURE,   

    ).to(DEVICE)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    print(f"\nOptimizer: AdamW(lr={LR}, weight_decay={WEIGHT_DECAY})")
    print(f"  Optimizing {sum(p.numel() for p in trainable_params):,} parameters")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, threshold=1e-3, min_lr=1e-7,
    )

    #  Sanity check 
    print("\n=== Sanity check (forward+backward on 1 sample) ===")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    sample = next(iter(val_loader))
    sulc_subj_sample = sample["sulc_subj"].to(DEVICE)
    wm_verts_sample = sample["wm_verts"].to(DEVICE)
    B = sulc_subj_sample.shape[0]
    sulc_subj_norm_sample = normalize(sulc_subj_sample)
    grid_for_model_sample = template_grid.unsqueeze(0).expand(B, -1, -1, -1)
    sphere_init_sample = template_sphere_init.expand(B, -1, -1)

    model.train()
    r_sample = model(
        sulc_subj=sulc_subj_norm_sample,
        template_grid=grid_for_model_sample,
        sphere_init=sphere_init_sample,
    )
    print(f"  Output shape: {tuple(r_sample.shape)}")
    n_out_sample = r_sample.shape[-1]

    loss_dict_sample = criterion(
        pred_verts=r_sample,
        sulc_subj=sulc_subj_norm_sample[:, :, :n_out_sample],
        template_grid=grid_for_model_sample,
        topology=model.topologies[model.ico_out],
        ref_verts_metric=wm_verts_sample[:, :, :n_out_sample],
        ref_verts_area=sphere_init_sample[:, :, :n_out_sample],
    )
    print(f"  sulc   = {loss_dict_sample['sulc'].item():.4f}")
    print(f"  metric = {loss_dict_sample['metric'].item():.4f}")
    
   
    if "area" in loss_dict_sample:
        print(f"  area   = {loss_dict_sample['area'].item():.4e}    "
            f"(weighted: {W_AREA * loss_dict_sample['area'].item():.4f})")
    if "count" in loss_dict_sample:
        print(f"  count  = {loss_dict_sample['count'].item():.4e}    "
            f"(weighted: {W_COUNT * loss_dict_sample['count'].item():.4f})")

    loss_dict_sample["total"].backward()
    optimizer.zero_grad()

    if torch.cuda.is_available():
        print(f"  Peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
    print("=== Sanity check passed ===\n")


    train_total_hist, train_sulc_hist, train_metric_hist = [], [], []
    #train_fold_hist = []
    #train_fold_area_hist = []
    #train_fold_count_hist = []  
    train_area_hist = []
    train_count_hist = [] 
    val_epochs = []
    val_total_hist, val_sulc_hist, val_metric_hist = [], [], []
    #val_fold_hist = []
    #val_fold_area_hist = []
    #val_fold_count_hist = []
    val_folding_hist = []   
    val_area_hist = []
    val_count_hist = []

    best_val_total = float('inf')
    epochs_without_improvement = 0   
    best_epoch = 0                    

    val_stats_epochs = []
    val_total_mean_hist = []
    val_total_std_hist = []
    val_best_subject_hist = []
    val_worst_subject_hist = []

    # Faces for folding counter
    faces_for_count = model.topologies[model.ico_out].faces.long().to(DEVICE)
    
    val_summary_csv = save_dir / "val_subject_summary_every_5_epochs.csv"

    with open(val_summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "val_total_mean",
            "val_total_std",
            "best_dataset",
            "best_subject",
            "best_total",
            "worst_dataset",
            "worst_subject",
            "worst_total",
        ])


    #  Training loop 
    for epoch in range(1, N_EPOCHS + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, template_grid, template_sphere_init, optimizer
        )
        scheduler.step(train_metrics["total"])

        train_total_hist.append(train_metrics["total"])
        train_sulc_hist.append(W_SULC * train_metrics["sulc"])
        train_metric_hist.append(W_METRIC * train_metrics.get("metric", 0.0))
        train_area_hist.append(W_AREA * train_metrics.get("area", 0.0))
        train_count_hist.append(W_COUNT * train_metrics.get("count", 0.0))   # NUOVO

        do_val = True
        if do_val:
            val_metrics = run_epoch(
                model, val_loader, criterion, template_grid, template_sphere_init, None
            )
            scheduler.step(val_metrics["total"])
            val_epochs.append(epoch)
            val_total_hist.append(val_metrics["total"])
            val_sulc_hist.append(W_SULC * val_metrics["sulc"])
            val_metric_hist.append(W_METRIC * val_metrics.get("metric", 0.0))
            #val_fold_hist.append(W_FOLD * val_metrics.get("fold", 0.0))
            #val_fold_area_hist.append(val_metrics.get("fold_area", 0.0))
            #val_fold_count_hist.append(val_metrics.get("fold_count", 0.0))
            val_area_hist.append(W_AREA * val_metrics.get("area", 0.0))
            val_count_hist.append(W_COUNT * val_metrics.get("count", 0.0))   # NUOVO
            diagnostic_forward(model, val_loader, template_grid, template_sphere_init)

            # Folding count 
            folding_pct = count_folded_full(
                model, val_loader, template_grid, template_sphere_init, faces_for_count
            )
            val_folding_hist.append(folding_pct)

            t_str = "  ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
            v_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            print(f"Epoch {epoch:3d}  lr={optimizer.param_groups[0]['lr']:.2e}")
            print(f"  train:    {t_str}")
            print(f"  val:      {v_str}")
            print(f"  folding%: {folding_pct:.4f}%  on val")

            # Early stopping check
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\n=== EARLY STOPPING at epoch {epoch} ===")
                print(f"    Best val_total: {best_val_total:.4f} (epoch {best_epoch})")
                print(f"    No improvement for {epochs_without_improvement} validation steps")
                break

            # Save best with early stopping
            if val_metrics["total"] < best_val_total - EARLY_STOPPING_MIN_DELTA:
                best_val_total = val_metrics["total"]
                best_epoch = epoch
                epochs_without_improvement = 0     # reset counter
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "folding_pct": folding_pct,
                }, save_dir / "best_val_total.pt")
                print(f"  >>> New best val_total: {best_val_total:.4f}")
            else:
                epochs_without_improvement += 1
                print(f"  No improvement for {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} validation steps")


        else:
            print(f"Epoch {epoch:3d}  lr={optimizer.param_groups[0]['lr']:.2e}  "
                  f"train_total={train_metrics['total']:.4f}  "
                  f"sulc={train_metrics['sulc']:.4f}  "
                  f"metric={train_metrics.get('metric',0):.4f}"
                  f"area={train_metrics.get('area',0):.4e}  "
                  f"count={train_metrics.get('count',0):.4e}")
                  

        do_val_subject_stats = (epoch % 5 == 0) or (epoch == 1) or (epoch == N_EPOCHS)

        if do_val_subject_stats:
            summary, subject_rows = evaluate_val_subjects(
                model,
                val_stats_loader,
                criterion,
                template_grid,
                template_sphere_init,
            )

            val_stats_epochs.append(epoch)
            val_total_mean_hist.append(summary["val_total_mean"])
            val_total_std_hist.append(summary["val_total_std"])
            val_best_subject_hist.append(summary["best_subject"])
            val_worst_subject_hist.append(summary["worst_subject"])

            print(
                f"  val subject stats @ epoch {epoch}: "
                f"mean={summary['val_total_mean']:.4f}, "
                f"std={summary['val_total_std']:.4f}, "
                f"best={summary['best_dataset']}/{summary['best_subject']} "
                f"({summary['best_total']:.4f}), "
                f"worst={summary['worst_dataset']}/{summary['worst_subject']} "
                f"({summary['worst_total']:.4f})"
            )

            with open(val_summary_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    summary["val_total_mean"],
                    summary["val_total_std"],
                    summary["worst_dataset"],
                    summary["worst_subject"],
                    summary["worst_total"],
                ])

            subject_csv = save_dir / f"val_subject_losses_epoch_{epoch:03d}.csv"
            with open(subject_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=subject_rows[0].keys())
                writer.writeheader()
                writer.writerows(subject_rows)

        if epoch % SAVE_EVERY == 0 or epoch == N_EPOCHS:
            ckpt_path = save_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
            }, ckpt_path)
            print(f"  Checkpoint -> {ckpt_path}")
            save_predicted_spheres(
                model, val_loader, template_grid, template_sphere_init, epoch, save_dir
            )
            model.train()


    
    hist_csv = save_dir / "loss_history.csv"
    with open(hist_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_total", "train_sulc", "train_metric",
                         "train_area", "train_count"])
        for i in range(len(train_total_hist)):
            writer.writerow([i + 1, train_total_hist[i], train_sulc_hist[i],
                             train_metric_hist[i], train_area_hist[i],
                             train_count_hist[i]])

    # Plot 

    epochs = range(1, len(train_total_hist) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        train_total_hist,
        label="train total"
    )

    plt.plot(
        val_epochs,
        val_total_hist,
        label="val total",
        linestyle="--",
        marker="o"
    )

  

    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.title("Total loss with validation subject variability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "loss_total.png", dpi=300)
    plt.close()

    # Loss components

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_sulc_hist,   label=f"train_sulc (x{W_SULC})",   color="C0")
    plt.plot(epochs, train_metric_hist, label=f"train_metric (x{W_METRIC})", color="C1")
    plt.plot(epochs, train_area_hist,  label=f"train_area (x{W_AREA})",  color="C2")
    plt.plot(epochs, train_count_hist,  label=f"train_count",  color="C3")   # NUOVO
    plt.plot(val_epochs, val_sulc_hist,   label=f"val_sulc",   color="C0", linestyle="--", marker="o")
    plt.plot(val_epochs, val_metric_hist, label=f"val_metric", color="C1", linestyle="--", marker="o")
    #plt.plot(val_epochs, val_fold_hist, label=f"val_fold", color="C2", linestyle="--", marker="o")
    plt.plot(val_epochs, val_area_hist,   label=f"val_area",   color="C2", linestyle="--", marker="o")
    plt.plot(val_epochs, val_count_hist,  label=f"val_count",  color="C3", linestyle="--", marker="o")   # NUOVO
    plt.xlabel("Epoch")
    plt.ylabel("Weighted loss contribution")
    plt.title(f"Loss components (ico=4 fine-tuning, FROZEN ico=3)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "loss_components.png")
    plt.close()

    # Plot del folding %
    plt.figure(figsize=(10, 6))
    plt.plot(val_epochs, val_folding_hist, marker="o", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Folded triangles (%)")
    plt.title("Folding % on validation set during training")
    plt.grid(True)
    plt.axhline(y=1.0, color="gray", linestyle=":", label="1% threshold")
    plt.axhline(y=0.1, color="green", linestyle=":", label="0.1% target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "folding_pct.png")
    plt.close()

    print(f"\nDone. Saved to: {save_dir}")
    print(f"Best val_total: {best_val_total:.4f}")
    if val_folding_hist:
        print(f"Final folding%: {val_folding_hist[-1]:.4f}%")


if __name__ == "__main__":
    main()