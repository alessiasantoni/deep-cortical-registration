"""
Training script for spherical registration with boxplot visualization.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from experiments.sphere_reg2.dataset_wm import (
    SphereRegDataset,
    load_template_grid,
    sample_grid_at_vertices,
)
#from experiments.sphere_reg2.network_norotation import SphereRegNet
from experiments.sphere_reg2.network_rotation import SphereRegNet

#from experiments.sphere_reg2.loss_wm import SulcalAlignmentLoss
from experiments.sphere_reg2.loss_totale import SulcalAlignmentLoss


# Configuration

DATA_ROOT  = "/projects/brainnet-data/sphere-reg"
GRID_DIR   = "experiments/sphere_reg2/fsaverage_grid"
DATASETS   = ["OASIS3"]  # lista dei dataset da includere (come appaiono nei file salvati)
HEMI       = "lh"

ICO_IN     = 3
ICO_OUT    = 3
N_ITER     = [15]    
BASE_CH    = 64
STEP_SIZE  = 0.01    
SPHERE_RADIUS = 100.0

N_EPOCHS   = 40    
BATCH_SIZE = 1
#LR         = 1e-4 # Projection
LR         = 5e-5  # Rotation 
GRAD_CLIP  = 0.5    

W_SULC          = 1.0
W_METRIC        = 0.5
W_AREA = 0.001

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = f"experiments/sphere_reg2/runs/{timestamp}"
SAVE_EVERY = 10
N_SAVE     = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# Sphere saving

def save_predicted_spheres(
    model, loader, template_grid, template_sphere_init, epoch, save_dir, n_save=N_SAVE,
):
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
                dataset = batch["dataset"][i]
                stem = out_dir / f"{dataset}_{subject}_{HEMI}"
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
            print(f"{'order':>6} {'iter':>5} {'verts':>8} {'|dr|_mean':>10} {'|dr|_max':>10} {'dr_std':>10}")
            for s in stats:
                print(f"{s['order']:>6} {s['iter']:>5} {s['n_verts']:>8} "
                      f"{s['dr_mean']:>10.4f} {s['dr_max']:>10.4f} {s['dr_std']:>10.4f}")
            print()
            break
    if was_training:
        model.train()

# Per-subject boxplot

def plot_per_subject_boxplot(per_subject, epoch, save_dir, n_highlight=5):
    """Boxplot dei loss per soggetto, evidenziando i n_highlight migliori e peggiori."""
    if not per_subject:
        return
    keys = ["total", "sulc", "metric", "area"]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 6))

    # Ordina per total loss per identificare best/worst
    sorted_subj = sorted(per_subject, key=lambda s: s["total"])
    best = sorted_subj[:n_highlight]
    worst = sorted_subj[-n_highlight:]

    for ax, key in zip(axes, keys):
        values = [s[key] for s in per_subject]
        ax.boxplot(values, vert=True, showfliers=True)
        # scatter di tutti i punti
        xs = np.random.normal(1, 0.04, size=len(values))
        ax.scatter(xs, values, alpha=0.4, s=20, color="gray", label="all")
        # highlight best
        best_vals = [s[key] for s in best]
        ax.scatter(np.random.normal(1, 0.04, size=len(best_vals)),
                   best_vals, color="green", s=60, label=f"best {n_highlight}", zorder=3)
        # highlight worst
        worst_vals = [s[key] for s in worst]
        ax.scatter(np.random.normal(1, 0.04, size=len(worst_vals)),
                   worst_vals, color="red", s=60, label=f"worst {n_highlight}", zorder=3)
        ax.set_title(f"{key} (epoch {epoch})")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        if key == "total":
            ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out_path = save_dir / f"boxplot_per_subject_epoch_{epoch:03d}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()

    # Salva anche un txt con la lista best/worst
    txt_path = save_dir / f"best_worst_epoch_{epoch:03d}.txt"
    with open(txt_path, "w") as f:
        f.write(f"=== Best {n_highlight} (lowest total loss) ===\n")
        for s in best:
            f.write(f"  {s['dataset']}/{s['subject']}  total={s['total']:.4f}  "
                    f"sulc={s['sulc']:.4f}  metric={s['metric']:.4f}  area={s['area']:.4f}\n")
        f.write(f"\n=== Worst {n_highlight} (highest total loss) ===\n")
        for s in worst:
            f.write(f"  {s['dataset']}/{s['subject']}  total={s['total']:.4f}  "
                    f"sulc={s['sulc']:.4f}  metric={s['metric']:.4f}  area={s['area']:.4f}\n")

    print(f"  Boxplot -> {out_path}")
    print(f"  Best/worst list -> {txt_path}")


# One epoch

def run_epoch_per_subject(model, loader, criterion, template_grid, template_sphere_init):
    """Come run_epoch ma in eval-only e restituisce loss per ogni soggetto."""
    model.eval()
    per_subject = []  # lista di dict: {subject, dataset, total, sulc, metric, area}

    with torch.no_grad():
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

            # Loss per-sample: itera sui sample del batch
            for i in range(B):
                loss_dict = criterion(
                    pred_verts=r[i:i+1],
                    sulc_subj=sulc_subj_norm[i:i+1, :, :n_out],
                    template_grid=grid_for_model[i:i+1],
                    topology=model.topologies[model.ico_out],
                    ref_verts_metric=wm_verts[i:i+1, :, :n_out],
                    ref_verts_area=sphere_init[i:i+1, :, :n_out],
                )
                if torch.isnan(loss_dict["total"]):
                    continue
                per_subject.append({
                    "subject": batch["subject"][i],
                    "dataset": batch["dataset"][i],
                    "total": loss_dict["total"].item(),
                    "sulc":  loss_dict["sulc"].item(),
                    "metric": loss_dict.get("metric", torch.tensor(0.0)).item(),
                    "area":  loss_dict.get("area",  torch.tensor(0.0)).item(),
                })
    return per_subject

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
            wm_verts = batch["wm_verts"].to(DEVICE)   # NUOVO: WM per metric loss
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

            # Printing per batch < 3
            #if batch_count < 3:

                #print(
                    #f"area raw = {loss_dict['area'].item():.12e}, "
                    #f"metric = {loss_dict['metric'].item():.12e}, "
                    #f"sulc = {loss_dict['sulc'].item():.12e}"
                #)
                
                #sulc_temp_pred = sample_grid_at_vertices(grid_for_model, r)
                #losses_str = "  ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items())
                #print(
                    #f"  [batch {batch_count}] "
                    #f"r_norm={r.norm(dim=1).mean().item():.3f}  "
                    #f"sulc_temp_std={sulc_temp_pred.std().item():.4f}  "
                    #f"{losses_str}"
                #)

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
                    if batch_count < 3:
                        print(f"  [batch {batch_count}] grad_norm before clip: {total_norm_before:.3f}")
                    optimizer.step()

            for k, v in loss_dict.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            n_valid_batches += 1
            batch_count += 1

    n_valid = max(n_valid_batches, 1)
    return {k: v / n_valid for k, v in sums.items()}


# Main

def main():
    print(f"Device: {DEVICE}")
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading template grid ...")
    template_grid = load_template_grid(GRID_DIR, HEMI).to(DEVICE)
    print(f"  template grid shape: {tuple(template_grid.shape)}")
    template_grid = standardize_template_grid(template_grid, HEMI)

    print("Loading template sphere init ...")
    template_sphere_init = get_template_sphere_init(HEMI, SPHERE_RADIUS, DEVICE)
    print(f"  template_sphere_init shape: {tuple(template_sphere_init.shape)}")

    dataset = SphereRegDataset(
        root_dir=DATA_ROOT,
        datasets=DATASETS,
        hemi=HEMI,
    )
    print(dataset)

    n_val = max(1, int(0.20 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

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
    print(f"SphereRegNet  ico_in={ICO_IN}  ico_out={ICO_OUT}  n_iter={N_ITER}  params={n_params:,}")

    criterion = SulcalAlignmentLoss(
        w_sulc=W_SULC,
        w_metric=W_METRIC,
        w_area=W_AREA,
        #w_folding=W_FOLDING,
        #folding_margin=FOLDING_MARGIN,
    ).to(DEVICE)

    print(f"Loss weights: sulc={W_SULC}, metric={W_METRIC}, area={W_AREA}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,        # dimezza il LR
        patience=5,        # aspetta 5 epoche senza miglioramento
        threshold=1e-3,    # miglioramento minimo per "contare"
        min_lr=1e-7,
    )

    train_total_hist, train_sulc_hist, train_metric_hist, train_area_hist = [], [], [], []
    val_epochs, val_total_hist, val_sulc_hist = [], [], []

    for epoch in range(1, N_EPOCHS + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, template_grid, template_sphere_init, optimizer
        )
        scheduler.step(train_metrics["total"])
        train_total_hist.append(train_metrics["total"])
        train_sulc_hist.append(train_metrics["sulc"])
        train_metric_hist.append(W_METRIC * train_metrics.get("metric", 0.0))
        train_area_hist.append(W_AREA * train_metrics.get("area", 0.0))

        if epoch % 5 == 0 or epoch == 1:
            val_metrics = run_epoch(
                model, val_loader, criterion, template_grid, template_sphere_init, None
            )
            val_epochs.append(epoch)
            val_total_hist.append(val_metrics["total"])
            val_sulc_hist.append(val_metrics["sulc"])
            diagnostic_forward(model, val_loader, template_grid, template_sphere_init)

            t_str = "  ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
            v_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            print(f"Epoch {epoch:3d}  train: {t_str}")
            print(f"          val:   {v_str}")
        else:
            print(f"Epoch {epoch:3d}  train_total={train_metrics['total']:.4f}  "
                  f"sulc={train_metrics['sulc']:.4f}  area={train_metrics.get('area',0):.4f}  "
                  f"metric={train_metrics.get('metric',0):.4f}")

        # NUOVO: boxplot e best/worst solo all'ultima epoca
        if epoch == N_EPOCHS:
            print("Computing per-subject losses for final boxplot ...")
            per_subject = run_epoch_per_subject(
                model, val_loader, criterion, template_grid, template_sphere_init
            )
            plot_per_subject_boxplot(per_subject, epoch, save_dir)          
        # =================================================

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
            save_predicted_spheres(
                model, val_loader, template_grid, template_sphere_init, epoch, save_dir
            )
            model.train()

    # Plot
    # Plot total loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, N_EPOCHS + 1), train_total_hist, label="train_total")
    plt.plot(val_epochs, val_total_hist, label="val_total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "loss_total.png")
    plt.close()


# Plot training components
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, N_EPOCHS + 1), train_sulc_hist, label="train_sulc")
    plt.plot(range(1, N_EPOCHS + 1), train_metric_hist, label="train_metric")
    plt.plot(range(1, N_EPOCHS + 1), train_area_hist, label="train_area")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss components")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "loss_components_train.png")
    plt.close()


    print(f"Saved loss plots in: {save_dir}")


if __name__ == "__main__":
    main()