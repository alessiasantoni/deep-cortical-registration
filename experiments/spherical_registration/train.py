"""
Training ico=3 from scratch, uniform sampling, with the variable depth
already validated previously.

Per-step logging: every train/val step appends one row to
`loss_per_step.csv`, with both the raw (unweighted) and weighted value of
each loss component, plus an `ico_level` column -- so that future work can
concatenate this CSV with the equivalent ones from ico4/5/6 training and
plot each loss component continuously across levels.
"""

from pathlib import Path

import argparse
import csv
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

from experiments.spherical_registration.dataset import (
    SphereRegDataset,
    load_template_grid,
    sample_grid_at_vertices,
)
from experiments.spherical_registration.network import SphereRegNet
from experiments.spherical_registration.loss import SulcalAlignmentLoss


# ============================================================
# CONFIG
# ============================================================

SPLIT_DIR = "/projects/brainnet-data/subject_splits"
BAD_SURFACES = "/projects/brainnet-data/subject_splits/bad_surfaces.json"

DATA_ROOT  = "/projects/brainnet-data/sphere-reg"
GRID_DIR   = "experiments/spherical_registration/fsaverage_grid"
DATASETS = ["ABIDE", "ADHD200", "ADNI3", "AIBL", "Buckner40",
            "Chinese-HCP", "HCP", "OASIS3", "ISBI2015"]
HEMI       = "lh"

ICO_LEVEL  = 3   # this training stage -- stored in every row of the per-step CSV
ICO_IN     = 3
ICO_OUT    = 3
BASE_CH    = 32
STEP_SIZE  = 0.01
SPHERE_RADIUS = 100.0

MIN_ITER = 15
MAX_ITER = 40
VAL_REFERENCE_ITER = 25

N_EPOCHS   = 25
BATCH_SIZE = 1
LR         = 1e-5
GRAD_CLIP  = 0.5
WEIGHT_DECAY = 1e-4

W_SULC     = 1.0
W_METRIC   = 0.5
W_AREA     = 0.1
W_COUNT    = 0.0
COUNT_TEMPERATURE = 1.0
FOLDED_REDUCTION = "sum"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = "runs"   # default; can be changed with --out_dir
SAVE_EVERY = 5
N_SAVE     = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-4

LOSS_COMPONENTS = ["sulc", "metric", "area", "count"]
COMPONENT_WEIGHTS = {"sulc": W_SULC, "metric": W_METRIC, "area": W_AREA, "count": W_COUNT}


# ============================================================
# Helpers
# ============================================================

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


# ============================================================
# Per-step loss CSV
# ============================================================

STEP_LOG_FIELDS = (
    ["ico_level", "epoch", "global_step", "split", "depth", "dataset", "subject"]
    + [f"{c}_raw" for c in LOSS_COMPONENTS]
    + [f"{c}_weighted" for c in LOSS_COMPONENTS]
    + ["total"]
)


def open_step_log(path):
    f = open(path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=STEP_LOG_FIELDS)
    writer.writeheader()
    return f, writer


def log_step(writer, log_file, epoch, global_step, split, depth, dataset, subject, loss_dict):
    row = {
        "ico_level": ICO_LEVEL, "epoch": epoch, "global_step": global_step,
        "split": split, "depth": depth, "dataset": dataset, "subject": subject,
        "total": loss_dict["total"].item(),
    }
    for c in LOSS_COMPONENTS:
        raw = loss_dict[c].item()
        row[f"{c}_raw"] = raw
        row[f"{c}_weighted"] = COMPONENT_WEIGHTS[c] * raw
    writer.writerow(row)
    log_file.flush()


# ============================================================
# Sphere saving
# ============================================================

def save_predicted_spheres(model, loader, template_grid, template_sphere_init,
                            epoch, save_dir, n_save=N_SAVE):
    out_dir = save_dir / "spheres" / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    faces_np = model.topologies[model.ico_out].faces.cpu().numpy().astype(np.int32)

    original_iter = model.n_iter[model.ico_in]
    model.n_iter[model.ico_in] = VAL_REFERENCE_ITER

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

            r = model(sulc_subj=sulc_subj_norm, template_grid=grid_for_model, sphere_init=sphere_init)

            for i in range(min(B, n_save - saved)):
                subject = batch["subject"][i]
                dataset_name = batch["dataset"][i]
                stem = out_dir / f"{dataset_name}_{subject}_{HEMI}"
                pred_np = r[i].T.cpu().numpy().astype(np.float32)
                nib.freesurfer.write_geometry(str(stem) + ".sphere.reg", pred_np, faces_np)
                saved += 1

    model.n_iter[model.ico_in] = original_iter
    print(f"  Saved {saved} sphere(s) -> {out_dir}")


# ============================================================
# Run epoch -- variable depth, NO rotation
# ============================================================

def run_epoch(model, loader, criterion, template_grid, template_sphere_init,
              optimizer, epoch, step_state, step_writer, step_log_file,
              min_iter=None, max_iter=None, fixed_iter=None):
    is_train = optimizer is not None
    split = "train" if is_train else "val"
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    sums = {}
    n_valid_batches = 0
    sampled_depths = []

    with ctx:
        for batch in loader:
            sulc_subj = batch["sulc_subj"].to(DEVICE)
            wm_verts = batch["wm_verts"].to(DEVICE)
            B = sulc_subj.shape[0]
            assert B == 1

            if is_train:
                depth = int(np.random.randint(min_iter, max_iter + 1))
                sampled_depths.append(depth)
            else:
                depth = fixed_iter
            model.n_iter[model.ico_in] = depth

            sulc_subj_norm = normalize(sulc_subj)
            grid_for_model = template_grid.unsqueeze(0).expand(B, -1, -1, -1)
            sphere_init = template_sphere_init.expand(B, -1, -1)

            r = model(sulc_subj=sulc_subj_norm, template_grid=grid_for_model, sphere_init=sphere_init)
            n_out = r.shape[-1]

            loss_dict = criterion(
                pred_verts=r,
                sulc_subj=sulc_subj_norm[:, :, :n_out],
                template_grid=grid_for_model,
                topology=model.topologies[model.ico_out],
                ref_verts_metric=wm_verts[:, :, :n_out],
            )

            if torch.isnan(loss_dict["total"]) or torch.isnan(r).any():
                if is_train:
                    optimizer.zero_grad()
                continue

            if is_train:
                optimizer.zero_grad()
                loss_dict["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=GRAD_CLIP
                )
                has_nan_grad = any(
                    p.grad is not None and torch.isnan(p.grad).any()
                    for p in model.parameters() if p.requires_grad
                )
                if has_nan_grad:
                    optimizer.zero_grad()
                    continue
                optimizer.step()

            step_state[split] += 1
            log_step(
                step_writer, step_log_file, epoch, step_state[split], split, depth,
                batch["dataset"][0], batch["subject"][0], loss_dict,
            )

            for k, v in loss_dict.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            n_valid_batches += 1

    n_valid = max(n_valid_batches, 1)
    result = {k: v / n_valid for k, v in sums.items()}
    if sampled_depths:
        result["_depth_mean"] = float(np.mean(sampled_depths))
    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train the ico3 network.")
    parser.add_argument("--out_dir", default=OUT_DIR,
                        help="root folder for outputs; results go in <out_dir>/ico3_train")
    args = parser.parse_args()

    save_dir = Path(args.out_dir) / "ico3_train"
    print(f"Device: {DEVICE}")
    print(f"Save dir: {save_dir}")

    save_dir.mkdir(parents=True, exist_ok=True)

    template_grid = load_template_grid(GRID_DIR, HEMI).to(DEVICE)
    template_grid = standardize_template_grid(template_grid, HEMI)
    template_sphere_init = get_template_sphere_init(HEMI, SPHERE_RADIUS, DEVICE)

    train_ds = SphereRegDataset(root_dir=DATA_ROOT, datasets=DATASETS, hemi=HEMI,
                                split_dir=SPLIT_DIR, split="train", bad_surfaces_file=BAD_SURFACES)
    val_ds   = SphereRegDataset(root_dir=DATA_ROOT, datasets=DATASETS, hemi=HEMI,
                                split_dir=SPLIT_DIR, split="validation", bad_surfaces_file=BAD_SURFACES)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SphereRegNet(
        ico_in=ICO_IN, ico_out=ICO_OUT, n_iter=[VAL_REFERENCE_ITER], in_channels=2,
        base_channels=BASE_CH, sphere_radius=SPHERE_RADIUS, step_size=STEP_SIZE,
    ).to(DEVICE)

    criterion = SulcalAlignmentLoss(
        w_sulc=W_SULC, w_metric=W_METRIC, w_area=W_AREA,
        folded_reduction=FOLDED_REDUCTION, w_count=W_COUNT,
        count_temperature=COUNT_TEMPERATURE,
    ).to(DEVICE)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, threshold=1e-3, min_lr=1e-7,
    )

    # --- Per-step CSV: raw + weighted value of every loss component, every step ---
    step_log_file, step_writer = open_step_log(save_dir / "loss_per_step.csv")
    step_state = {"train": 0, "val": 0}

    # --- Per-epoch history, for the summary plots (raw AND weighted) ---
    train_total_hist, val_total_hist, val_epochs = [], [], []
    train_raw_hist = {c: [] for c in LOSS_COMPONENTS}
    train_weighted_hist = {c: [] for c in LOSS_COMPONENTS}
    val_raw_hist = {c: [] for c in LOSS_COMPONENTS}
    val_weighted_hist = {c: [] for c in LOSS_COMPONENTS}

    best_val_total = float('inf')
    epochs_without_improvement = 0

    for epoch in range(1, N_EPOCHS + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, template_grid, template_sphere_init,
            optimizer, epoch, step_state, step_writer, step_log_file,
            min_iter=MIN_ITER, max_iter=MAX_ITER,
        )
        scheduler.step(train_metrics["total"])
        train_total_hist.append(train_metrics["total"])
        for c in LOSS_COMPONENTS:
            train_raw_hist[c].append(train_metrics[c])
            train_weighted_hist[c].append(COMPONENT_WEIGHTS[c] * train_metrics[c])

        val_metrics = run_epoch(
            model, val_loader, criterion, template_grid, template_sphere_init,
            optimizer=None, epoch=epoch, step_state=step_state,
            step_writer=step_writer, step_log_file=step_log_file,
            fixed_iter=VAL_REFERENCE_ITER,
        )
        scheduler.step(val_metrics["total"])
        val_epochs.append(epoch)
        val_total_hist.append(val_metrics["total"])
        for c in LOSS_COMPONENTS:
            val_raw_hist[c].append(val_metrics[c])
            val_weighted_hist[c].append(COMPONENT_WEIGHTS[c] * val_metrics[c])

        print(f"Epoch {epoch:3d}  train_total={train_metrics['total']:.4f} "
              f"(depth mean={train_metrics.get('_depth_mean', float('nan')):.1f})  "
              f"val_total(depth={VAL_REFERENCE_ITER})={val_metrics['total']:.4f}")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"EARLY STOPPING at epoch {epoch}")
            break

        if val_metrics["total"] < best_val_total - EARLY_STOPPING_MIN_DELTA:
            best_val_total = val_metrics["total"]
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "min_iter": MIN_ITER, "max_iter": MAX_ITER,
                "val_reference_iter": VAL_REFERENCE_ITER,
            }, save_dir / "best_val_total.pt")
            print(f"  >>> New best: {best_val_total:.4f}")
        else:
            epochs_without_improvement += 1

        if epoch % SAVE_EVERY == 0 or epoch == N_EPOCHS:
            save_predicted_spheres(model, val_loader, template_grid, template_sphere_init,
                                    epoch, save_dir)
            model.train()

    step_log_file.close()
    print(f"\nPer-step loss log -> {save_dir / 'loss_per_step.csv'}")

    # ============================================================
    # Plots (English)
    # ============================================================

    epochs_range = range(1, len(train_total_hist) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_total_hist, label="train")
    plt.plot(val_epochs, val_total_hist, label="val", linestyle="--", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.legend()
    plt.grid(True)
    plt.title("ico3 -- total loss")
    plt.tight_layout()
    plt.savefig(save_dir / "loss_total.png", dpi=200)
    plt.close()

    colors = {"sulc": "C0", "metric": "C1", "area": "C2", "count": "C3"}

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax = axes[0]
    for c in LOSS_COMPONENTS:
        ax.plot(epochs_range, train_raw_hist[c], label=f"train_{c}", color=colors[c])
        ax.plot(val_epochs, val_raw_hist[c], label=f"val_{c}", color=colors[c],
                linestyle="--", marker="o")
    ax.set_ylabel("Raw (unweighted) loss")
    ax.set_title("ico3 -- loss components, raw")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for c in LOSS_COMPONENTS:
        ax.plot(epochs_range, train_weighted_hist[c], label=f"train_{c} (x{COMPONENT_WEIGHTS[c]})",
                color=colors[c])
        ax.plot(val_epochs, val_weighted_hist[c], label=f"val_{c}", color=colors[c],
                linestyle="--", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted loss contribution")
    ax.set_title("ico3 -- loss components, weighted (contribution to total)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "loss_components.png", dpi=200)
    plt.close()

    print(f"\nBest val_total: {best_val_total:.4f}")


if __name__ == "__main__":
    main()
