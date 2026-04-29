from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from brainnet.networks.toporeg import TopoReg

from experiments.sphere_reg.dataset import (
    SphereRegDataset,
    load_feature_grid_images,
)
from experiments.sphere_reg.sulcal_loss import SulcalAlignmentLoss

#############################################################################################
# le due funzioni seguenti implementano una rotazione 3D random dei valori sulc, funzionando come augmentation dei dati

#from scipy.spatial import cKDTree
#def random_rotation_matrix() -> torch.Tensor:
#    x = np.random.uniform(size=3)
#    theta = 2 * np.pi * x[0]
#    phi   = 2 * np.pi * x[1]
#    z     = x[2]
#    V = np.array([np.cos(phi) * np.sqrt(z), np.sin(phi) * np.sqrt(z), np.sqrt(1 - z)])
#    H = np.eye(3) - 2 * np.outer(V, V)
#    R = H @ np.array([[np.cos(theta), np.sin(theta), 0],
#                       [-np.sin(theta), np.cos(theta), 0],
#                       [0, 0, 1]])
#    return torch.tensor(R, dtype=torch.float32)

#def rotate_sulc(sulc: torch.Tensor, ico_verts: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
#    verts_np = ico_verts.cpu().numpy()
#    verts_rotated = (R.numpy() @ verts_np.T).T
#    tree = cKDTree(verts_np)
#    _, idx = tree.query(verts_rotated, k=1, workers=-1)
#    idx = torch.from_numpy(idx.astype(np.int64))
#    return sulc[:, :, idx]
#############################################################################################


DATA_ROOT  = "/projects/brainnet-data/sphere-reg"
GRID_DIR   = "experiments/sphere_reg/fsaverage_grid"
DATASETS   = ["Buckner40"]
#DATASETS   = ["COBRE"]
HEMI       = "lh"

IN_ORDER   = 3   # coarsest scale
OUT_ORDER  = 5   
MAX_ORDER  = 6    # topology hierarchy depth built inside TopoReg

N_EPOCHS   = 30
BATCH_SIZE = 1 #4
ACCUM_STEPS = 4
LR         = 1e-3

# Loss weights
W_SULC   = 1.0    # sulcal depth alignment (the main training signal)
W_AREA   = 1.0   # face-area regularisation
W_METRIC = 0.0   # edge-length regularisation

# Saving
SAVE_DIR   = "experiments/sphere_reg/runs/sulcal_reg"
SAVE_EVERY = 10   # save spheres + checkpoint every N epochs
N_SAVE     = 5    # how many validation subjects to save per checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helpers

def surfaces_to_verts(out: dict, hemi: str) -> torch.Tensor:
    """Extract (B, 3, N) from TopoReg output {"sphere.reg": {hemi: Surface}}."""
    return out["sphere.reg"][hemi].vertices.transpose(1, 2)  # (B, N, 3) -> (B, 3, N)


def expand_grid(t: torch.Tensor, B: int, device: torch.device) -> torch.Tensor:
    """Expand (1, H, W) template grid to (B, 1, H, W) on device."""
    return t.to(device).unsqueeze(0).expand(B, -1, -1, -1)


def normalize_grid(grid: torch.Tensor) -> torch.Tensor:
    mean = grid.mean(dim=(-2, -1), keepdim=True)
    std = grid.std(dim=(-2, -1), keepdim=True) + 1e-8
    return (grid - mean) / std


# Sphere saving

def save_registered_spheres(
    model: TopoReg,
    #model: TopoRegNative,
    loader: DataLoader,
    template_grids: dict,
    epoch: int,
    save_dir: Path,
    n_save: int = N_SAVE,
) -> None:
    """Run model on the first n_save validation subjects and write FreeSurfer
    geometry files so the registration can be inspected in FreeView.

    Saved files
    -----------
    {hemi}.sphere.reg   — predicted (registered) sphere positions
    {hemi}.sphere.input — input template sphere positions (identity baseline)
    """
    out_dir = save_dir / "spheres" / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    faces_np = model.out_topology[HEMI].faces.cpu().numpy().astype(np.int32)

    model.eval()
    saved = 0
    with torch.no_grad():
        for batch in loader:
            if saved >= n_save:
                break

            sulc_subj = batch["sulc_subj"][HEMI].to(DEVICE)   # (B, 1, N)
            B = sulc_subj.shape[0]
            sulc_temp_grid = expand_grid(template_grids[HEMI], B, DEVICE)

            sulc_subj_norm = (sulc_subj - sulc_subj.mean()) / (sulc_subj.std() + 1e-8)
            sulc_temp_norm = normalize_grid(sulc_temp_grid)
            out = model(sulc_subj={HEMI: sulc_subj_norm}, sulc_temp={HEMI: sulc_temp_norm})
            pred_verts = out["sphere.reg"][HEMI].vertices  # (B, N, 3)

            # For a fair comparison use the OUT_ORDER template positions
            tmpl_verts_out = model.r_temp(HEMI, OUT_ORDER).T.cpu().numpy()  # (N, 3)

            for i in range(min(B, n_save - saved)):
                subject  = batch["subject"][i]
                dataset  = batch["dataset"][i]
                stem     = out_dir / f"{dataset}_{subject}_{HEMI}"

                pred_np = pred_verts[i].cpu().numpy().astype(np.float32)

                # Predicted registered sphere
                nib.freesurfer.write_geometry(
                    str(stem) + ".sphere.reg",
                    pred_np,
                    faces_np,
                )
                # Input (template) sphere for comparison
                nib.freesurfer.write_geometry(
                    str(stem) + ".sphere.input",
                    tmpl_verts_out.astype(np.float32),
                    faces_np,
                )
                saved += 1

    print(f"  Saved {saved} sphere(s) → {out_dir}")




# One epoch

def run_epoch(
    model: TopoReg,
    loader: DataLoader,
    criterion: SulcalAlignmentLoss,
    template_grids: dict,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    sums: dict[str, float] = {}

    with ctx:
        for batch in loader:
            sulc_subj = batch["sulc_subj"][HEMI].to(DEVICE)      # (B, 1, N)
            ico_verts = batch["ico_verts"][HEMI].to(DEVICE)       # (B, 3, N)
            B = sulc_subj.shape[0]
            sulc_temp_grid = expand_grid(template_grids[HEMI], B, DEVICE)  # (B, 1, H, W)

            # Normalise subject sulc (1D) and template grid (2D)
            sulc_subj_norm = (sulc_subj - sulc_subj.mean()) / (sulc_subj.std() + 1e-8)
            sulc_temp_norm = normalize_grid(sulc_temp_grid)

            out = model(
                sulc_subj={HEMI: sulc_subj_norm},     # mesh (B, 1, N)
                sulc_temp={HEMI: sulc_temp_norm},     # grid (B, 1, H, W)
            )

            pred_verts = surfaces_to_verts(out, HEMI)
            topology   = model.out_topology[HEMI]
            pred_verts_normalized = F.normalize(pred_verts, dim=1)


            loss_dict = criterion(
                pred_verts=pred_verts_normalized,
                sulc_subj=sulc_subj_norm,         
                sulc_temp_grid=sulc_temp_norm,
                topology=topology,
                ref_verts=F.normalize(ico_verts, dim=1),
            )




            # DEBUG esteso
            if torch.isnan(loss_dict["total"]):
                print(f"NaN in loss!")
                print(f"  pred_verts: min={pred_verts.min().item()}, max={pred_verts.max().item()}")
                print(f"  pred_verts has NaN: {torch.isnan(pred_verts).any().item()}")
                print(f"  sulc_subj norm: min={sulc_subj_norm.min().item()}, max={sulc_subj_norm.max().item()}")
                print(f"  sulc_temp norm: min={sulc_temp_norm.min().item()}, max={sulc_temp_norm.max().item()}")
                for k, v in loss_dict.items():
                    print(f"  {k}: {v.item()}")
                import sys; sys.exit(1)


            if is_train:
                optimizer.zero_grad()
                loss_dict["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

            for k, v in loss_dict.items():
                sums[k] = sums.get(k, 0.0) + v.item()

    n = len(loader)
    return {k: v / n for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)


    # Template sulcal depth (from pre-computed fsaverage spherical grid)
  
    print("Loading template sulcal depth grid ...")
    template_grids = load_feature_grid_images(
        grid_dir=GRID_DIR,
        hemispheres=(HEMI,),
        feature="sulc",
    )
    print(f"  {HEMI}: shape={tuple(template_grids[HEMI].shape)}")

    # Dataset 

    dataset = SphereRegDataset(
        root_dir=DATA_ROOT,
        datasets=DATASETS,
        hemispheres=(HEMI,),
        ico_order=OUT_ORDER,
        template_sulc=None,
    )
    print(dataset)


    n_val   = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


    # Model

    model = TopoReg(
        in_order=IN_ORDER,
        out_order=OUT_ORDER,
        max_order=MAX_ORDER,
        in_channels=1,
        encoder_channels=32,
        reg_channels=dict(encoder=[64, 64, 64, 64], decoder=[64, 64, 64]),
        n_steps=5, ##########################
        sphere_radius=100.0,
    ).to(DEVICE)




    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TopoReg  in_order={IN_ORDER}  out_order={OUT_ORDER}  params={n_params:,}")

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    
    #ico_positions = dataset._ico_verts[HEMI]  # (N_ico, 3)
    #ico_positions = full_dataset._ico_verts[HEMI]
    #print(f"ico_positions shape: {ico_positions.shape}")
    #print(f"ico_positions device: {ico_positions.device}")

    criterion = SulcalAlignmentLoss(
        w_sulc=W_SULC,
        w_area=W_AREA,
        w_metric=W_METRIC,
    ).to(DEVICE)

  
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

  
    # Training loop
   

    # ------------------- stampa loss plots ----------------------------
    train_total_hist = []
    train_sulc_hist = []
    train_area_hist = []
    train_metric_hist = []

    val_epochs = []
    val_total_hist = []
    val_sulc_hist = []
    val_area_hist = []
    val_metric_hist = []
    # ------------------------------------------------------------------

    for epoch in range(1, N_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, criterion, template_grids, optimizer)

        train_total_hist.append(train_metrics["total"])
        train_sulc_hist.append(train_metrics["sulc"])
        train_area_hist.append(train_metrics["area"])
        train_metric_hist.append(train_metrics["metric"])

        if epoch % 5 == 0 or epoch == 1:
            val_metrics = run_epoch(model, val_loader, criterion, template_grids, None)

            val_epochs.append(epoch)
            val_total_hist.append(val_metrics["total"])
            val_sulc_hist.append(val_metrics["sulc"])
            val_area_hist.append(val_metrics["area"])
            val_metric_hist.append(val_metrics["metric"])

            t_str = "  ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
            v_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            print(f"Epoch {epoch:3d}  train: {t_str}")
            print(f"          val:   {v_str}")
        else:
            print(f"Epoch {epoch:3d}  train_total={train_metrics['total']:.4f}")

        if epoch % SAVE_EVERY == 0 or epoch == N_EPOCHS:
            # Save model checkpoint
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
            print(f"  Checkpoint → {ckpt_path}")

            # Save registered spheres for visual inspection
            save_registered_spheres(model, val_loader, template_grids, epoch, save_dir)
            model.train()


   
    # Plot losses
  
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

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, N_EPOCHS + 1), train_sulc_hist, label="train_sulc")
    plt.plot(val_epochs, val_sulc_hist, label="val_sulc")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Sulc loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "loss_sulc.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, N_EPOCHS + 1), train_area_hist, label="train_area")
    plt.plot(val_epochs, val_area_hist, label="val_area")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Area loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "loss_area.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, N_EPOCHS + 1), train_metric_hist, label="train_metric")
    plt.plot(val_epochs, val_metric_hist, label="val_metric")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Metric loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "loss_metric.png")
    plt.close()

    print(f"Saved loss plots in: {save_dir}")


if __name__ == "__main__":
    main()


# Per eliminare sfere dei training precedenti
#rm -rf experiments/sphere_reg/runs/sulcal_reg/spheres
#rm -rf experiments/sphere_reg/runs/sulcal_reg/*.pt

# Per eliminare plot delle loss functions
# rm experiments/sphere_reg/runs/sulcal_reg/loss_*.png