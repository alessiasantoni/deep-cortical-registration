"""
Second stage: adds one finer level per run (--level 4, 5 or 6) on top of the
previous one. The weights of the lower levels come from
<out_dir>/ico{level-1}_train/best_val_total.pt and stay frozen, only the new
level is trained.

As in train.py, the number of ico3 iterations is random while training (range read
from the checkpoint) and fixed for validation. Same step CSV as train.py, plus
loss_history.csv with one row per epoch and the % of folded triangles.

  python -m experiments.spherical_registration.finetune_train --level 4 [--out_dir runs]

DATA_ROOT, SPLIT_DIR and BAD_SURFACES below are also used by the evaluation scripts.
"""

import argparse
from pathlib import Path

import csv
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from experiments.spherical_registration.dataset import (
    SphereRegDataset,
    load_template_grid,
    sample_grid_at_vertices,
)
from experiments.spherical_registration.network import SphereRegNet
from experiments.spherical_registration.loss import SulcalAlignmentLoss, compute_signed_face_areas


SPLIT_DIR = "/projects/brainnet-data/subject_splits"
BAD_SURFACES = "/projects/brainnet-data/subject_splits/bad_surfaces.json"

DATA_ROOT  = "/projects/brainnet-data/sphere-reg"
GRID_DIR   = "experiments/spherical_registration/fsaverage_grid"
DATASETS = ["ABIDE", "ADHD200", "ADNI3", "AIBL", "Buckner40",
            "Chinese-HCP", "HCP", "OASIS3", "ISBI2015"]
HEMI       = "lh"

SAVE_ROOT = Path("runs")   # replaced by --out_dir in main()

ICO_IN     = 3
BASE_CH    = 32
STEP_SIZE  = 0.01
SPHERE_RADIUS = 100.0

# iterations at each finer level (the ico3 ones are read from the checkpoint)
FINE_N_ITER = {4: 7, 5: 5, 6: 3}

N_EPOCHS   = 25
BATCH_SIZE = 1
LR         = 1e-4
GRAD_CLIP  = 0.5
WEIGHT_DECAY = 1e-4

# loss weights of each level; metric grows with the level to limit distortion
LEVEL_WEIGHTS = {
    4: {"sulc": 1.0, "metric": 0.6, "area": 0.005, "count": 1e-4},
    5: {"sulc": 1.0, "metric": 1.0, "area": 0.005, "count": 1e-6},
    6: {"sulc": 1.0, "metric": 2.0, "area": 0.005, "count": 1e-6},
}
COUNT_TEMPERATURE = 1.0
FOLDED_REDUCTION = "sum"

SAVE_EVERY = 5
N_SAVE     = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-4

LOSS_COMPONENTS = ["sulc", "metric", "area", "count"]


def stage_dir(level: int) -> Path:
    return SAVE_ROOT / f"ico{level}_train"


def previous_checkpoint(level: int) -> Path:
    return stage_dir(level - 1) / "best_val_total.pt"


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


def folded_percent(pred_verts: torch.Tensor, faces: torch.Tensor) -> float:
    """Exact % of triangles with negative signed area. pred_verts: (B, 3, N), faces: (F, 3)."""
    signed_area = compute_signed_face_areas(pred_verts.transpose(1, 2), faces)
    return 100.0 * (signed_area < 0).float().mean().item()


def get_coarse_depth_config(ckpt):
    """Range of ico3 iterations used in training and the value used for validation.
    Old checkpoints may not have them, hence the fallbacks."""
    min_iter, max_iter = ckpt.get("min_iter"), ckpt.get("max_iter")
    val_ref = ckpt.get("val_reference_iter")
    if min_iter is None or max_iter is None:
        fallback = val_ref if val_ref is not None else 15
        print(f"  no min_iter/max_iter in checkpoint, using {fallback}")
        min_iter = max_iter = fallback
    if val_ref is None:
        val_ref = (min_iter + max_iter) // 2
        print(f"  no val_reference_iter in checkpoint, using {val_ref}")
    print(f"  ico3 iterations: [{min_iter}, {max_iter}], {val_ref} in validation")
    return min_iter, max_iter, val_ref


def load_lower_weights(model: SphereRegNet, ckpt):
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    # strict=False: the weights of the new level are not in the checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  {len(missing)} new parameters, {len(unexpected)} unexpected")


def freeze_lower_subnets(model, freeze_orders):
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(f"unets.{o}." in name or f"deforms.{o}." in name for o in freeze_orders):
            param.requires_grad = False
            frozen_count += param.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  frozen levels {list(freeze_orders)}: {frozen_count:,} frozen, "
          f"{trainable:,} trainable, {total:,} total")


def build_stage_model(level, prev_ckpt):
    """Network from ico3 to `level`, lower levels loaded and frozen."""
    coarse_min, coarse_max, val_depth = get_coarse_depth_config(prev_ckpt)
    n_iter = [val_depth] + [FINE_N_ITER[l] for l in range(ICO_IN + 1, level + 1)]
    model = SphereRegNet(
        ico_in=ICO_IN, ico_out=level, n_iter=n_iter, in_channels=2,
        base_channels=BASE_CH, sphere_radius=SPHERE_RADIUS, step_size=STEP_SIZE,
    ).to(DEVICE)
    load_lower_weights(model, prev_ckpt)
    freeze_lower_subnets(model, list(range(ICO_IN, level)))
    return model, coarse_min, coarse_max, val_depth


# same columns as in train.py
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


def log_step(writer, log_file, level, weights, epoch, global_step, split, depth,
             dataset, subject, loss_dict):
    row = {
        "ico_level": level, "epoch": epoch, "global_step": global_step,
        "split": split, "depth": depth, "dataset": dataset, "subject": subject,
        "total": loss_dict["total"].item(),
    }
    for c in LOSS_COMPONENTS:
        raw = loss_dict[c].item()
        row[f"{c}_raw"] = raw
        row[f"{c}_weighted"] = weights[c] * raw
    writer.writerow(row)
    log_file.flush()


def save_predicted_spheres(model, loader, template_grid, template_sphere_init,
                            epoch, save_dir, val_depth, n_save=N_SAVE):
    out_dir = save_dir / "spheres" / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    faces_np = model.topologies[model.ico_out].faces.cpu().numpy().astype(np.int32)

    original_iter = model.n_iter[model.ico_in]
    model.n_iter[model.ico_in] = val_depth

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


def run_epoch(model, loader, criterion, template_grid, template_sphere_init,
              optimizer, epoch, level, weights, step_state, step_writer, step_log_file,
              depth_min=None, depth_max=None, fixed_depth=None):
    is_train = optimizer is not None
    split = "train" if is_train else "val"
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    sums = {}
    n_valid_batches = 0
    sampled_depths = []
    folded_pcts = []
    faces = model.topologies[model.ico_out].faces.long().to(DEVICE)

    with ctx:
        for batch in loader:
            sulc_subj = batch["sulc_subj"].to(DEVICE)
            wm_verts = batch["wm_verts"].to(DEVICE)
            B = sulc_subj.shape[0]
            assert B == 1

            if is_train:
                depth = int(np.random.randint(depth_min, depth_max + 1))
                sampled_depths.append(depth)
            else:
                depth = fixed_depth
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
                step_writer, step_log_file, level, weights, epoch, step_state[split],
                split, depth, batch["dataset"][0], batch["subject"][0], loss_dict,
            )
            folded_pcts.append(folded_percent(r.detach(), faces))

            for k, v in loss_dict.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            n_valid_batches += 1

    n_valid = max(n_valid_batches, 1)
    result = {k: v / n_valid for k, v in sums.items()}
    if sampled_depths:
        result["_depth_mean"] = float(np.mean(sampled_depths))
    result["_folded_pct"] = float(np.mean(folded_pcts)) if folded_pcts else float("nan")
    return result


def train_level(level, train_loader, val_loader, template_grid, template_sphere_init):
    weights = LEVEL_WEIGHTS[level]
    save_dir = stage_dir(level)
    save_dir.mkdir(parents=True, exist_ok=True)
    prev_ckpt_path = previous_checkpoint(level)

    print(f"\nico{level}: weights {weights}")
    print(f"starting from {prev_ckpt_path}, saving in {save_dir}")

    prev_ckpt = torch.load(prev_ckpt_path, map_location=DEVICE)
    model, coarse_min, coarse_max, val_depth = build_stage_model(level, prev_ckpt)

    criterion = SulcalAlignmentLoss(
        w_sulc=weights["sulc"], w_metric=weights["metric"], w_area=weights["area"],
        folded_reduction=FOLDED_REDUCTION, w_count=weights["count"],
        count_temperature=COUNT_TEMPERATURE,
    ).to(DEVICE)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, threshold=1e-3, min_lr=1e-7,
    )

    step_log_file, step_writer = open_step_log(save_dir / "loss_per_step.csv")
    step_state = {"train": 0, "val": 0}

    # train_<c>/val_<c> are the weighted values, *_raw the unweighted ones
    history_fields = (
        ["epoch", "train_total"] + [f"train_{c}" for c in LOSS_COMPONENTS]
        + ["val_total"] + [f"val_{c}" for c in LOSS_COMPONENTS]
        + [f"train_{c}_raw" for c in LOSS_COMPONENTS]
        + [f"val_{c}_raw" for c in LOSS_COMPONENTS]
        + ["train_folded_pct", "val_folded_pct"]
    )
    history_path = save_dir / "loss_history.csv"
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(history_fields)

    train_total_hist, val_total_hist, val_epochs = [], [], []
    train_raw_hist = {c: [] for c in LOSS_COMPONENTS}
    train_weighted_hist = {c: [] for c in LOSS_COMPONENTS}
    val_raw_hist = {c: [] for c in LOSS_COMPONENTS}
    val_weighted_hist = {c: [] for c in LOSS_COMPONENTS}
    train_folded_hist, val_folded_hist = [], []

    best_val_total = float('inf')
    epochs_without_improvement = 0

    for epoch in range(1, N_EPOCHS + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, template_grid, template_sphere_init,
            optimizer, epoch, level, weights, step_state, step_writer, step_log_file,
            depth_min=coarse_min, depth_max=coarse_max,
        )
        scheduler.step(train_metrics["total"])
        train_total_hist.append(train_metrics["total"])
        train_folded_hist.append(train_metrics["_folded_pct"])
        for c in LOSS_COMPONENTS:
            train_raw_hist[c].append(train_metrics[c])
            train_weighted_hist[c].append(weights[c] * train_metrics[c])

        val_metrics = run_epoch(
            model, val_loader, criterion, template_grid, template_sphere_init,
            None, epoch, level, weights, step_state, step_writer, step_log_file,
            fixed_depth=val_depth,
        )
        scheduler.step(val_metrics["total"])
        val_epochs.append(epoch)
        val_total_hist.append(val_metrics["total"])
        val_folded_hist.append(val_metrics["_folded_pct"])
        for c in LOSS_COMPONENTS:
            val_raw_hist[c].append(val_metrics[c])
            val_weighted_hist[c].append(weights[c] * val_metrics[c])

        print(f"Epoch {epoch:3d}  train {train_metrics['total']:.4f}  val {val_metrics['total']:.4f}  "
              f"folded {train_folded_hist[-1]:.3f}% / {val_folded_hist[-1]:.3f}%")

        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, train_total_hist[-1]]
                + [train_weighted_hist[c][-1] for c in LOSS_COMPONENTS]
                + [val_total_hist[-1]]
                + [val_weighted_hist[c][-1] for c in LOSS_COMPONENTS]
                + [train_raw_hist[c][-1] for c in LOSS_COMPONENTS]
                + [val_raw_hist[c][-1] for c in LOSS_COMPONENTS]
                + [train_folded_hist[-1], val_folded_hist[-1]]
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

        if val_metrics["total"] < best_val_total - EARLY_STOPPING_MIN_DELTA:
            best_val_total = val_metrics["total"]
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "min_iter": coarse_min, "max_iter": coarse_max,
                "val_reference_iter": val_depth,
            }, save_dir / "best_val_total.pt")
            print(f"  best so far ({best_val_total:.4f}), checkpoint saved")
        else:
            epochs_without_improvement += 1

        if epoch % SAVE_EVERY == 0 or epoch == N_EPOCHS:
            save_predicted_spheres(model, val_loader, template_grid, template_sphere_init,
                                    epoch, save_dir, val_depth)
            model.train()

    step_log_file.close()
    print(f"\nPer-step loss log -> {save_dir / 'loss_per_step.csv'}")

    epochs_range = range(1, len(train_total_hist) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_total_hist, label="train")
    plt.plot(val_epochs, val_total_hist, label="val", linestyle="--", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.legend()
    plt.grid(True)
    plt.title(f"ico{level} -- total loss")
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
    ax.set_title(f"ico{level} -- loss components, raw")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for c in LOSS_COMPONENTS:
        ax.plot(epochs_range, train_weighted_hist[c], label=f"train_{c} (x{weights[c]})",
                color=colors[c])
        ax.plot(val_epochs, val_weighted_hist[c], label=f"val_{c}", color=colors[c],
                linestyle="--", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted loss contribution")
    ax.set_title(f"ico{level} -- loss components, weighted (contribution to total)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "loss_components.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_folded_hist, label="train")
    plt.plot(val_epochs, val_folded_hist, label="val", linestyle="--", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Folded triangles (%)")
    plt.title(f"ico{level} -- folded triangles (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "folding_pct.png", dpi=200)
    plt.close()

    print(f"\nico{level} best val_total: {best_val_total:.4f}")


def main():
    global SAVE_ROOT
    parser = argparse.ArgumentParser(description="Fine-tune one ico level (4, 5 or 6).")
    parser.add_argument("--level", type=int, required=True, choices=sorted(FINE_N_ITER),
                        help="level to fine-tune; starts from the previous level's best checkpoint")
    parser.add_argument("--out_dir", default=str(SAVE_ROOT),
                        help="root folder for outputs; results go in <out_dir>/ico<level>_train")
    args = parser.parse_args()
    SAVE_ROOT = Path(args.out_dir)

    print(f"Device: {DEVICE}")

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

    train_level(args.level, train_loader, val_loader, template_grid, template_sphere_init)


if __name__ == "__main__":
    main()
