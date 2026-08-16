"""
Training ico=3 with PRE-COMPUTED rotation (deterministic, estimated via
grid search on the sulc loss, +-20 degrees), replacing the previous
random augmentation. Each subject ALWAYS uses its own estimated rotation,
both in training and in validation (it's a preprocessing step, not noise).
"""

from pathlib import Path

import csv
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial.transform import Rotation as R

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
HEMI       = "lh"

# Rotation lookup
ROTATION_LOOKUP_CSV = Path(
    "/home/s252125/my_project/experiments/sphere_reg2/runs/precomputed_rotations/rotation_lookup.csv"
)

ICO_IN     = 3
ICO_OUT    = 3
N_ITER     = [15]
BASE_CH    = 32
STEP_SIZE  = 0.01
SPHERE_RADIUS = 100.0

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
SAVE_DIR = f"experiments/sphere_reg2/runs/ico3_precomputed_rot"
SAVE_EVERY = 5
N_SAVE     = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-4


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



def load_rotation_lookup(csv_path, device, dtype=torch.float32):
    lookup = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], row["subject"])
            angles = [float(row["alpha"]), float(row["beta"]), float(row["gamma"])]
            rot = R.from_euler("ZYZ", angles, degrees=True)
            R_mat = torch.tensor(rot.as_matrix(), dtype=dtype, device=device)
            R_inv_mat = torch.tensor(rot.inv().as_matrix(), dtype=dtype, device=device)
            lookup[key] = (R_mat, R_inv_mat)
    print(f"Rotation lookup loaded: {len(lookup)} subjects")
    return lookup


def get_rotation_for_batch(rotation_lookup, dataset_name, subject_name, device, dtype):
    key = (dataset_name, subject_name)
    if key in rotation_lookup:
        return rotation_lookup[key]
    print(f"  WARNING: rotation not found for {key}, using identity")
    identity = torch.eye(3, dtype=dtype, device=device)
    return identity, identity



def save_predicted_spheres(model, loader, template_grid, template_sphere_init,
                            rotation_lookup, epoch, save_dir, n_save=N_SAVE):
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

            R_mat, _ = get_rotation_for_batch(
                rotation_lookup, batch["dataset"][0], batch["subject"][0],
                sphere_init.device, sphere_init.dtype
            )
            sphere_init_rot = torch.einsum("ij,bjn->bin", R_mat, sphere_init)

            r = model(
                sulc_subj=sulc_subj_norm,
                template_grid=grid_for_model,
                sphere_init=sphere_init_rot,
            )
            n_out = r.shape[-1]

            for i in range(min(B, n_save - saved)):
                subject = batch["subject"][i]
                dataset_name = batch["dataset"][i]
                stem = out_dir / f"{dataset_name}_{subject}_{HEMI}"
                pred_np = r[i].T.cpu().numpy().astype(np.float32)
                init_np = sphere_init_rot[i, :, :n_out].T.cpu().numpy().astype(np.float32)
                nib.freesurfer.write_geometry(str(stem) + ".sphere.reg", pred_np, faces_np)
                nib.freesurfer.write_geometry(str(stem) + ".sphere.input", init_np, faces_np)
                saved += 1

    print(f"  Saved {saved} sphere(s) -> {out_dir}")



def run_epoch(model, loader, criterion, template_grid, template_sphere_init,
              rotation_lookup, optimizer):
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
            assert B == 1, "This scheme assumes BATCH_SIZE=1 (one rotation per batch)"

            sulc_subj_norm = normalize(sulc_subj)
            grid_for_model = template_grid.unsqueeze(0).expand(B, -1, -1, -1)
            sphere_init = template_sphere_init.expand(B, -1, -1)

            # pre-computed rotation for this subject
            R_mat, R_inv_mat = get_rotation_for_batch(
                rotation_lookup, batch["dataset"][0], batch["subject"][0],
                sphere_init.device, sphere_init.dtype
            )
            sphere_init_input = torch.einsum("ij,bjn->bin", R_mat, sphere_init)

            r = model(
                sulc_subj=sulc_subj_norm,
                template_grid=grid_for_model,
                sphere_init=sphere_init_input,
            )
            n_out = r.shape[-1]

            # de-rotation before loss computation
            r = torch.einsum("ij,bjn->bin", R_inv_mat, r)

            loss_dict = criterion(
                pred_verts=r,
                sulc_subj=sulc_subj_norm[:, :, :n_out],
                template_grid=grid_for_model,
                topology=model.topologies[model.ico_out],
                ref_verts_metric=wm_verts[:, :, :n_out],
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
                    msg = (f"  [batch {batch_count}] "
                           f"sulc={loss_dict.get('sulc', torch.tensor(0.0)).item():.4f}  "
                           f"metric={loss_dict.get('metric', torch.tensor(0.0)).item():.4f}  ")
                    if "area" in loss_dict:
                        msg += f"area={loss_dict['area'].item():.4e}  "
                    print(msg)

                optimizer.step()

            for k, v in loss_dict.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            n_valid_batches += 1
            batch_count += 1

    n_valid = max(n_valid_batches, 1)
    return {k: v / n_valid for k, v in sums.items()}



def evaluate_val_subjects(model, loader, criterion, template_grid, template_sphere_init,
                           rotation_lookup):
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in loader:
            sulc_subj = batch["sulc_subj"].to(DEVICE)
            wm_verts = batch["wm_verts"].to(DEVICE)
            B = sulc_subj.shape[0]
            if B != 1:
                raise ValueError("evaluate_val_subjects expects BATCH_SIZE=1.")

            subject = batch["subject"][0]
            dataset_name = batch["dataset"][0]

            sulc_subj_norm = normalize(sulc_subj)
            grid_for_model = template_grid.unsqueeze(0).expand(B, -1, -1, -1)
            sphere_init = template_sphere_init.expand(B, -1, -1)

            R_mat, R_inv_mat = get_rotation_for_batch(
                rotation_lookup, dataset_name, subject, sphere_init.device, sphere_init.dtype
            )
            sphere_init_input = torch.einsum("ij,bjn->bin", R_mat, sphere_init)

            r = model(
                sulc_subj=sulc_subj_norm,
                template_grid=grid_for_model,
                sphere_init=sphere_init_input,
            )
            n_out = r.shape[-1]
            r = torch.einsum("ij,bjn->bin", R_inv_mat, r)

            loss_dict = criterion(
                pred_verts=r,
                sulc_subj=sulc_subj_norm[:, :, :n_out],
                template_grid=grid_for_model,
                topology=model.topologies[model.ico_out],
                ref_verts_metric=wm_verts[:, :, :n_out],
            )

            rows.append({
                "dataset": dataset_name, "subject": subject,
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
        "best_dataset": best_row["dataset"], "best_subject": best_row["subject"],
        "best_total": best_row["total"],
        "worst_dataset": worst_row["dataset"], "worst_subject": worst_row["subject"],
        "worst_total": worst_row["total"],
    }
    return summary, rows



def count_folded_full(model, loader, template_grid, template_sphere_init, rotation_lookup, faces):
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

            R_mat, R_inv_mat = get_rotation_for_batch(
                rotation_lookup, batch["dataset"][0], batch["subject"][0],
                sphere_init.device, sphere_init.dtype
            )
            sphere_init_input = torch.einsum("ij,bjn->bin", R_mat, sphere_init)

            r = model(sulc_subj=sulc_subj_norm, template_grid=grid_for_model,
                      sphere_init=sphere_init_input)
            r = torch.einsum("ij,bjn->bin", R_inv_mat, r)

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

    print("\nLoading rotation lookup ...")
    rotation_lookup = load_rotation_lookup(ROTATION_LOOKUP_CSV, DEVICE)

    train_ds = SphereRegDataset(root_dir=DATA_ROOT, datasets=DATASETS, hemi=HEMI,
                                split_dir=SPLIT_DIR, split="train",
                                bad_surfaces_file=BAD_SURFACES)
    val_ds   = SphereRegDataset(root_dir=DATA_ROOT, datasets=DATASETS, hemi=HEMI,
                                split_dir=SPLIT_DIR, split="validation",
                                bad_surfaces_file=BAD_SURFACES)

    print(f"\n{'='*50}\nTOTAL Train: {len(train_ds)}   TOTAL Val: {len(val_ds)}\n{'='*50}\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_stats_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    print(f"\nBuilding model: ICO_IN={ICO_IN}, ICO_OUT={ICO_OUT}, N_ITER={N_ITER}")
    model = SphereRegNet(
        ico_in=ICO_IN, ico_out=ICO_OUT, n_iter=N_ITER, in_channels=2,
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

    faces_for_count = model.topologies[model.ico_out].faces.long().to(DEVICE)

    train_total_hist, val_total_hist, val_epochs = [], [], []
    val_folding_hist = []
    best_val_total = float('inf')
    epochs_without_improvement = 0
    best_epoch = 0

    val_summary_csv = save_dir / "val_subject_summary_every_5_epochs.csv"
    with open(val_summary_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "val_total_mean", "val_total_std",
            "best_dataset", "best_subject", "best_total",
            "worst_dataset", "worst_subject", "worst_total",
        ])

    for epoch in range(1, N_EPOCHS + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, template_grid, template_sphere_init,
            rotation_lookup, optimizer
        )
        scheduler.step(train_metrics["total"])
        train_total_hist.append(train_metrics["total"])

        val_metrics = run_epoch(
            model, val_loader, criterion, template_grid, template_sphere_init,
            rotation_lookup, None
        )
        scheduler.step(val_metrics["total"])
        val_epochs.append(epoch)
        val_total_hist.append(val_metrics["total"])

        folding_pct = count_folded_full(
            model, val_loader, template_grid, template_sphere_init, rotation_lookup, faces_for_count
        )
        val_folding_hist.append(folding_pct)

        print(f"Epoch {epoch:3d}  lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"train_total={train_metrics['total']:.4f}  val_total={val_metrics['total']:.4f}  "
              f"folding%={folding_pct:.4f}%")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\n=== EARLY STOPPING at epoch {epoch} ===")
            print(f"    Best val_total: {best_val_total:.4f} (epoch {best_epoch})")
            break

        if val_metrics["total"] < best_val_total - EARLY_STOPPING_MIN_DELTA:
            best_val_total = val_metrics["total"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "folding_pct": folding_pct,
            }, save_dir / "best_val_total.pt")
            print(f"  >>> New best val_total: {best_val_total:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}")

        if (epoch % 5 == 0) or (epoch == 1) or (epoch == N_EPOCHS):
            summary, subject_rows = evaluate_val_subjects(
                model, val_stats_loader, criterion, template_grid, template_sphere_init,
                rotation_lookup,
            )
            print(f"  val subject stats @ epoch {epoch}: mean={summary['val_total_mean']:.4f}, "
                  f"worst={summary['worst_dataset']}/{summary['worst_subject']} ({summary['worst_total']:.4f})")
            with open(val_summary_csv, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch, summary["val_total_mean"], summary["val_total_std"],
                    summary["best_dataset"], summary["best_subject"], summary["best_total"],
                    summary["worst_dataset"], summary["worst_subject"], summary["worst_total"],
                ])

        if epoch % SAVE_EVERY == 0 or epoch == N_EPOCHS:
            ckpt_path = save_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
            }, ckpt_path)
            save_predicted_spheres(
                model, val_loader, template_grid, template_sphere_init,
                rotation_lookup, epoch, save_dir
            )
            model.train()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_total_hist) + 1), train_total_hist, label="train total")
    plt.plot(val_epochs, val_total_hist, label="val total", linestyle="--", marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Total loss")
    plt.title("Total loss (pre-computed rotation, +-20 degrees)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(save_dir / "loss_total.png", dpi=300)
    plt.close()

    print(f"\nDone. Saved to: {save_dir}")
    print(f"Best val_total: {best_val_total:.4f}")


if __name__ == "__main__":
    main()