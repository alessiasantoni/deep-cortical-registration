"""
Versione di train.py per debug:
  - allena su 5 soggetti totali (4 train + 1 val)
  - usa loss_smooth (continua, non discontinua come OrientedAreaLoss)
  - template stats hardcoded (dal prof)
  - plot di tutte le componenti della loss
  - salva le sfere SIA del training SIA del validation set ad ogni checkpoint
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from experiments.sphere_reg2.dataset import (
    SphereRegDataset,
    load_template_grid,
    sample_grid_at_vertices,
)
from experiments.sphere_reg2.network_norotation import SphereRegNet
from experiments.sphere_reg2.loss_smooth import SulcalAlignmentLoss


# ============== Configuration ==============

DATA_ROOT = "/projects/brainnet-data/sphere-reg"
GRID_DIR  = "experiments/sphere_reg/fsaverage_grid"
DATASETS  = ["ABIDE"]
HEMI      = "lh"

# === SOGGETTI SCELTI ===
TRAIN_SUBJECTS = ["sub-0210", "sub-0211", "sub-0212", "sub-0213"]
VAL_SUBJECTS   = ["sub-0215"]
TRAIN_DATASET  = "ABIDE"

# Hyperparameters
ICO_IN     = 3
ICO_OUT    = 6
N_ITER     = [3, 3, 3, 5]
BASE_CH    = 64
STEP_SIZE  = 0.1
SPHERE_RADIUS = 100.0

N_EPOCHS   = 50
BATCH_SIZE = 1
LR         = 5e-5
GRAD_CLIP  = 1.0

# Pesi loss
W_SULC     = 1.0
W_AREA     = 0.5
W_METRIC   = 0.5

# Debug verbose
DEBUG_VERBOSE = False

# Save dir
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = f"experiments/sphere_reg2/runs/debug5_{timestamp}"
SAVE_EVERY = 10
N_SAVE_VAL = 5
N_SAVE_TRAIN = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============== Template stats hardcoded (dal prof) ==============

TEMPLATE_SULC_STATS = {
    "lh": {"mean": -1.1523885e-08, "std": 0.59384775},
    "rh": {"mean": -4.1938106e-09, "std":  0.5924748},
}


# ============== Helpers ==============

def normalize(x: torch.Tensor, dims: tuple = None) -> torch.Tensor:
    if dims is None:
        dims = tuple(range(1, x.ndim))
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True) + 1e-8
    return (x - mean) / std


def standardize_template_grid(template_grid: torch.Tensor, hemi: str) -> torch.Tensor:
    """Standardize using hardcoded stats from Jesper."""
    stats = TEMPLATE_SULC_STATS[hemi]
    mean = stats["mean"]
    std = stats["std"]
    grid_norm = (template_grid - mean) / std
    print(f"  Template grid standardized (hardcoded): mean={mean:.4e}, std={std:.4f}")
    return grid_norm


def get_template_sphere_init(hemi: str, sphere_radius: float, device) -> torch.Tensor:
    import brainsynth.resources
    template_resources = brainsynth.resources.Template()
    template_sphere = template_resources.load_surface(hemi, "sphere.reg")["vertices"]
    template_sphere = template_sphere.T.unsqueeze(0).float().to(device)
    template_sphere = F.normalize(template_sphere, dim=1) * sphere_radius
    return template_sphere


def check_tensor(name, t):
    if t is None:
        print(f"  {name}: None")
        return
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    marker = "🚨" if (has_nan or has_inf) else "  "
    print(f"  {marker}{name:25s} shape={tuple(t.shape)} "
          f"min={t.min().item():.4f} max={t.max().item():.4f} "
          f"mean={t.mean().item():.4f} std={t.std().item():.4f} "
          f"NaN={has_nan} Inf={has_inf}")


# ============== Sphere saving ==============

def save_predicted_spheres(
    model, loader, template_grid, template_sphere_init, epoch, save_dir,
    n_save=5, prefix="val",
):
    """Salva le sfere predette dal modello.
    
    prefix: 'train' o 'val' — crea sottocartella separata per i due set.
    """
    out_dir = save_dir / "spheres" / f"epoch_{epoch:03d}" / prefix
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
                dataset = batch["dataset"][i]
                stem = out_dir / f"{dataset}_{subject}_{HEMI}"
                pred_np = r[i].T.cpu().numpy().astype(np.float32)
                init_np = sphere_init[i, :, :n_out].T.cpu().numpy().astype(np.float32)
                nib.freesurfer.write_geometry(str(stem) + ".sphere.reg", pred_np, faces_np)
                nib.freesurfer.write_geometry(str(stem) + ".sphere.input", init_np, faces_np)
                saved += 1

    print(f"  Saved {saved} sphere(s) [{prefix}] -> {out_dir}")


# ============== Diagnostic ==============

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
            print(f"{'order':>6} {'iter':>5} {'verts':>8} {'|dr|_mean':>10} {'|dr|_max':>10} {'dr_std':>10}")
            for s in stats:
                print(f"{s['order']:>6} {s['iter']:>5} {s['n_verts']:>8} "
                      f"{s['dr_mean']:>10.4f} {s['dr_max']:>10.4f} {s['dr_std']:>10.4f}")
            print()
            break
    if was_training:
        model.train()


# ============== Training loop ==============

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
                ref_verts=sphere_init[:, :, :n_out],
            )

            if DEBUG_VERBOSE or batch_count < 3:
                sulc_temp_pred = sample_grid_at_vertices(grid_for_model, r)
                if DEBUG_VERBOSE:
                    print(f"\n--- Batch {batch_count} | Subject: {batch['subject'][0]} ---")
                    check_tensor("sulc_subj_norm", sulc_subj_norm)
                    check_tensor("sphere_init", sphere_init)
                    check_tensor("r (after model)", r)
                    check_tensor("sulc_temp_pred", sulc_temp_pred)
                    for k, v in loss_dict.items():
                        check_tensor(f"loss_{k}", v.unsqueeze(0))
                else:
                    losses_str = "  ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items())
                    print(
                        f"  [batch {batch_count}] subject={batch['subject'][0]} "
                        f"r_norm={r.norm(dim=1).mean().item():.3f}  "
                        f"{losses_str}"
                    )

            if torch.isnan(loss_dict["total"]) or torch.isnan(r).any():
                print(f"  [batch {batch_count}] NaN in forward, subject: {batch['subject'][0]} - SKIPPING")
                if is_train:
                    optimizer.zero_grad()
                batch_count += 1
                continue

            if is_train:
                optimizer.zero_grad()
                loss_dict["total"].backward()

                total_norm_before = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=GRAD_CLIP
                )

                if DEBUG_VERBOSE:
                    n_with_nan = 0
                    worst_param = ""
                    worst_grad = 0.0
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            n_with_nan += 1
                        grad_max = p.grad.abs().max().item()
                        if grad_max > worst_grad:
                            worst_grad = grad_max
                            worst_param = name
                    print(f"  grad_norm before clip: {total_norm_before:.4e}")
                    print(f"  worst gradient: {worst_param} (max={worst_grad:.4e})")
                    print(f"  params with NaN/Inf grad: {n_with_nan}")

                has_nan_grad = any(
                    p.grad is not None and torch.isnan(p.grad).any()
                    for p in model.parameters()
                )

                if has_nan_grad:
                    print(f"  [batch {batch_count}] NaN gradients in subject: "
                          f"{batch['subject'][0]} ({batch['dataset'][0]}) - SKIPPING step")
                    optimizer.zero_grad()
                    batch_count += 1
                    continue
                else:
                    if not DEBUG_VERBOSE and batch_count < 3:
                        print(f"  [batch {batch_count}] grad_norm before clip: {total_norm_before:.4e}")
                    optimizer.step()

            for k, v in loss_dict.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            n_valid_batches += 1
            batch_count += 1

    n_valid = max(n_valid_batches, 1)
    return {k: v / n_valid for k, v in sums.items()}


# ============== Main ==============

def main():
    print(f"Device: {DEVICE}")
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save dir: {save_dir}")

    print("\nLoading template grid ...")
    template_grid = load_template_grid(GRID_DIR, HEMI).to(DEVICE)
    print(f"  template grid shape: {tuple(template_grid.shape)}")
    template_grid = standardize_template_grid(template_grid, HEMI)

    print("\nLoading template sphere init ...")
    template_sphere_init = get_template_sphere_init(HEMI, SPHERE_RADIUS, DEVICE)
    print(f"  template_sphere_init shape: {tuple(template_sphere_init.shape)}")

    full_dataset = SphereRegDataset(
        root_dir=DATA_ROOT,
        datasets=DATASETS,
        hemi=HEMI,
    )
    print(f"\nFull dataset: {full_dataset}")

    train_indices = []
    val_indices = []
    for idx, (ds, sub) in enumerate(full_dataset.subjects):
        if ds == TRAIN_DATASET and sub in TRAIN_SUBJECTS:
            train_indices.append(idx)
        elif ds == TRAIN_DATASET and sub in VAL_SUBJECTS:
            val_indices.append(idx)

    if len(train_indices) == 0:
        raise RuntimeError(f"No training subjects found! Check TRAIN_SUBJECTS/TRAIN_DATASET")
    if len(val_indices) == 0:
        raise RuntimeError(f"No val subjects found! Check VAL_SUBJECTS/TRAIN_DATASET")

    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)

    print(f"\n=== Selected subjects ===")
    print(f"Train ({len(train_ds)}):")
    for idx in train_indices:
        print(f"  - {full_dataset.subjects[idx]}")
    print(f"Val ({len(val_ds)}):")
    for idx in val_indices:
        print(f"  - {full_dataset.subjects[idx]}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Loader senza shuffle per salvare le sfere training in ordine costante
    train_loader_eval = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SphereRegNet(
        ico_in=ICO_IN,
        ico_out=ICO_OUT,
        n_iter=N_ITER,
        in_channels=2,
        base_channels=BASE_CH,
        sphere_radius=SPHERE_RADIUS,
        step_size=STEP_SIZE,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nSphereRegNet  ico_in={ICO_IN}  ico_out={ICO_OUT}  n_iter={N_ITER}  params={n_params:,}")

    criterion = SulcalAlignmentLoss(
        w_sulc=W_SULC,
        w_area=W_AREA,
        w_metric=W_METRIC,
    ).to(DEVICE)
    print(f"Loss weights: sulc={W_SULC}, area={W_AREA}, metric={W_METRIC}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # History per tutte le componenti
    train_total_hist, train_sulc_hist, train_area_hist, train_metric_hist = [], [], [], []
    val_epochs, val_total_hist, val_sulc_hist, val_area_hist, val_metric_hist = [], [], [], [], []

    for epoch in range(1, N_EPOCHS + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, template_grid, template_sphere_init, optimizer
        )
        train_total_hist.append(train_metrics["total"])
        train_sulc_hist.append(train_metrics["sulc"])
        train_area_hist.append(train_metrics.get("area", 0.0))
        train_metric_hist.append(train_metrics.get("metric", 0.0))

        if epoch % 5 == 0 or epoch == 1:
            val_metrics = run_epoch(
                model, val_loader, criterion, template_grid, template_sphere_init, None
            )
            val_epochs.append(epoch)
            val_total_hist.append(val_metrics["total"])
            val_sulc_hist.append(val_metrics["sulc"])
            val_area_hist.append(val_metrics.get("area", 0.0))
            val_metric_hist.append(val_metrics.get("metric", 0.0))
            diagnostic_forward(model, val_loader, template_grid, template_sphere_init)

            t_str = "  ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
            v_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            print(f"Epoch {epoch:3d}  train: {t_str}")
            print(f"          val:   {v_str}")
        else:
            print(f"Epoch {epoch:3d}  train_total={train_metrics['total']:.4f}  "
                  f"sulc={train_metrics['sulc']:.4f}  area={train_metrics.get('area',0):.4f}  "
                  f"metric={train_metrics.get('metric',0):.4f}")

        if epoch % SAVE_EVERY == 0 or epoch == N_EPOCHS:
            ckpt_path = save_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                },
                ckpt_path,
            )
            print(f"  Checkpoint -> {ckpt_path}")

            # Salva sfere VAL
            save_predicted_spheres(
                model, val_loader, template_grid, template_sphere_init,
                epoch, save_dir, n_save=N_SAVE_VAL, prefix="val",
            )
            # Salva sfere TRAIN
            save_predicted_spheres(
                model, train_loader_eval, template_grid, template_sphere_init,
                epoch, save_dir, n_save=N_SAVE_TRAIN, prefix="train",
            )
            model.train()

    # === Plot ===
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, N_EPOCHS + 1), train_total_hist, label="train_total")
    plt.plot(val_epochs, val_total_hist, label="val_total", marker='o')
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Total loss")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(save_dir / "loss_total.png"); plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, N_EPOCHS + 1), train_sulc_hist, label="train_sulc")
    plt.plot(range(1, N_EPOCHS + 1), train_area_hist, label="train_area")
    plt.plot(range(1, N_EPOCHS + 1), train_metric_hist, label="train_metric")
    plt.plot(val_epochs, val_sulc_hist, label="val_sulc", marker='o')
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss components")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(save_dir / "loss_components.png"); plt.close()

    print(f"\nSaved loss plots in: {save_dir}")


if __name__ == "__main__":
    main()