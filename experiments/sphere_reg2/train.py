""" 
Training script
launch with:
cd /home/s252125/my_project
python -m experiments.sphere_reg2.train

How spheres are saved after the registration:

experiments/sphere_reg2/runs/
├── checkpoint_epoch_010.pt          ← pesi del modello
├── checkpoint_epoch_020.pt
├── checkpoint_epoch_030.pt
├── loss_total.png                   ← plot della loss totale
├── loss_sulc.png                    ← plot della loss NCC
└── spheres/
    ├── epoch_010/
    │   ├── Buckner40_sub-XXX_lh.sphere.reg     ← predetta dal modello
    │   ├── Buckner40_sub-XXX_lh.sphere.input   ← sfera iniziale (riferimento)
    │   └── ... (5 soggetti)
    ├── epoch_020/
    └── epoch_030/

SAVE_EVERY is the number of epochs between saving model checkpoints and predicted spheres 
for visual inspection.

N_SAVE is the number of subjects to save for visual inspection at each checkpoint 

To download spheres on windows:
scp -r s252125@health-pcalc2:/home/s252125/my_project/experiments/sphere_reg2/runs/spheres C:\spheres_sulc_v2\

per salvare solo un soggetto, tutte le epochs:
scp -r s252125@health-pcalc2:/home/s252125/my_project/experiments/sphere_reg2/runs/spheres/*/*sub-0043* C:\spheres_sulc_v2\


""" 

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from experiments.sphere_reg2.dataset import (
    SphereRegDataset,
    load_template_grid,
)
from experiments.sphere_reg2.network import SphereRegNet
from experiments.sphere_reg2.loss import SulcalAlignmentLoss


DATA_ROOT  = "/projects/brainnet-data/sphere-reg"
GRID_DIR   = "experiments/sphere_reg/fsaverage_grid"
DATASETS   = ["Buckner40"]
HEMI       = "lh"

N_ITER     = 1
ICO_ORDER  = 6
BASE_CH    = 64
STEP_SIZE  = 0.1

N_EPOCHS   = 50
BATCH_SIZE = 1
LR         = 1e-5
GRAD_CLIP  = 0.5

SAVE_DIR   = "experiments/sphere_reg2/runs"
SAVE_EVERY = 10
N_SAVE     = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(x: torch.Tensor, dims: tuple = None) -> torch.Tensor:
    if dims is None:
        dims = tuple(range(1, x.ndim))
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True) + 1e-8
    return (x - mean) / std


def save_predicted_spheres(
    model: SphereRegNet,
    loader: DataLoader,
    template_grid: torch.Tensor,
    epoch: int,
    save_dir: Path,
    n_save: int = N_SAVE,
):
    out_dir = save_dir / "spheres" / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    faces_np = model.topologies[-1].faces.cpu().numpy().astype(np.int32)

    model.eval()
    saved = 0
    with torch.no_grad():
        for batch in loader:
            if saved >= n_save:
                break
            sulc_subj = batch["sulc_subj"].to(DEVICE)
            sphere_init = batch["sphere_reg"].to(DEVICE)
            B = sulc_subj.shape[0]

            sulc_subj_norm = normalize(sulc_subj)
            template_grid_norm = normalize(template_grid.unsqueeze(0).expand(B, -1, -1, -1))

            r = model(
                sulc_subj=sulc_subj_norm,
                template_grid=template_grid_norm,
                sphere_init=sphere_init,
            )

            for i in range(min(B, n_save - saved)):
                subject = batch["subject"][i]
                dataset = batch["dataset"][i]
                stem = out_dir / f"{dataset}_{subject}_{HEMI}"
                pred_np = r[i].T.cpu().numpy().astype(np.float32)
                init_np = sphere_init[i].T.cpu().numpy().astype(np.float32)
                nib.freesurfer.write_geometry(str(stem) + ".sphere.reg", pred_np, faces_np)
                nib.freesurfer.write_geometry(str(stem) + ".sphere.input", init_np, faces_np)
                saved += 1

    print(f"  Saved {saved} sphere(s) -> {out_dir}")


def run_epoch(
    model: SphereRegNet,
    loader: DataLoader,
    criterion: SulcalAlignmentLoss,
    template_grid: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    from experiments.sphere_reg2.dataset import sample_grid_at_vertices

    is_train = optimizer is not None
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    sums: dict[str, float] = {}
    batch_count = 0
    n_valid_batches = 0

    with ctx:
        for batch in loader:
            sulc_subj = batch["sulc_subj"].to(DEVICE)
            sphere_init = batch["sphere_reg"].to(DEVICE)
            B = sulc_subj.shape[0]

            sulc_subj_norm = normalize(sulc_subj)
            template_grid_norm = normalize(template_grid.unsqueeze(0).expand(B, -1, -1, -1))

            r = model(
                sulc_subj=sulc_subj_norm,
                template_grid=template_grid_norm,
                sphere_init=sphere_init,
            )

            loss_dict = criterion(
                pred_verts=r,
                sulc_subj=sulc_subj_norm,
                template_grid=template_grid_norm,
            )

            if batch_count < 3:
                sulc_temp_pred = sample_grid_at_vertices(template_grid_norm, r)
                print(
                    f"  [batch {batch_count}] "
                    f"pred_verts_norm={r.norm(dim=1).mean().item():.3f}  "
                    f"sulc_subj_std={sulc_subj_norm.std().item():.4f}  "
                    f"sulc_temp_pred_std={sulc_temp_pred.std().item():.4f}  "
                    f"loss={loss_dict['total'].item():.6f}"
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

                has_nan_grad = any(
                    p.grad is not None and torch.isnan(p.grad).any()
                    for p in model.parameters()
                )

                if has_nan_grad:
                    print(
                        f"  [batch {batch_count}] NaN gradients in subject: "
                        f"{batch['subject'][0]} ({batch['dataset'][0]}) - SKIPPING step"
                    )
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


def main():
    print(f"Device: {DEVICE}")
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading template grid ...")
    template_grid = load_template_grid(GRID_DIR, HEMI).to(DEVICE)
    print(f"  template grid shape: {tuple(template_grid.shape)}")

    dataset = SphereRegDataset(
        root_dir=DATA_ROOT,
        datasets=DATASETS,
        hemi=HEMI,
    )
    print(dataset)

    n_val = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SphereRegNet(
        n_iter=N_ITER,
        ico_order=ICO_ORDER,
        in_channels=2,
        base_channels=BASE_CH,
        sphere_radius=100.0,
        step_size=STEP_SIZE,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SphereRegNet  n_iter={N_ITER}  ico={ICO_ORDER}  params={n_params:,}")

    criterion = SulcalAlignmentLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    train_total_hist, train_sulc_hist = [], []
    val_epochs, val_total_hist, val_sulc_hist = [], [], []

    for epoch in range(1, N_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, criterion, template_grid, optimizer)
        train_total_hist.append(train_metrics["total"])
        train_sulc_hist.append(train_metrics["sulc"])

        if epoch % 5 == 0 or epoch == 1:
            val_metrics = run_epoch(model, val_loader, criterion, template_grid, None)
            val_epochs.append(epoch)
            val_total_hist.append(val_metrics["total"])
            val_sulc_hist.append(val_metrics["sulc"])

            t_str = "  ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
            v_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            print(f"Epoch {epoch:3d}  train: {t_str}")
            print(f"          val:   {v_str}")
        else:
            print(f"Epoch {epoch:3d}  train_total={train_metrics['total']:.4f}")

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
            save_predicted_spheres(model, val_loader, template_grid, epoch, save_dir)
            model.train()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, N_EPOCHS + 1), train_total_hist, label="train_total")
    plt.plot(val_epochs, val_total_hist, label="val_total")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Total loss")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(save_dir / "loss_total.png"); plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, N_EPOCHS + 1), train_sulc_hist, label="train_sulc")
    plt.plot(val_epochs, val_sulc_hist, label="val_sulc")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Sulc loss (NCC)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(save_dir / "loss_sulc.png"); plt.close()

    print(f"Saved loss plots in: {save_dir}")


if __name__ == "__main__":
    main()

