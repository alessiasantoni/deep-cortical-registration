"""Training script for TopoRegNative — coarse-to-fine registration at native resolution.

Works at the full fsaverage native resolution (245 762 vertices) as the finest
scale, going coarse-to-fine from ico2 (162 vertices) up through ico7
(163 842 vertices) and finally native (245 762 vertices).  No resampling is
needed: the dataset returns the raw native-resolution sulcal depth directly.

Usage
-----
    /home/s252125/miniforge3/envs/brainnet/bin/python \
        -m experiments.sphere_reg.train_toporeg_native
"""

import torch
from torch.utils.data import DataLoader, random_split

from brainnet.networks.toporeg import TopoRegNative

from experiments.sphere_reg.dataset import SphereRegDataset, compute_template_sulc
from experiments.sphere_reg.registration_loss import RegistrationLossFull

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT  = "/projects/brainnet-data/sphere-reg"
DATASETS   = ["Buckner40"]
HEMI       = "lh"

IN_ORDER   = 6    # 2 is the coarsest ico scale (162 vertices), 6 is 40962 vertices 
# out_order is always 7 for TopoRegNative, then native (245762) is added

N_EPOCHS   = 100
BATCH_SIZE = 2    # native resolution uses more memory than ico5
LR         = 1e-4

# Loss weights
W_SULC   = 1.0
W_AREA   = 1e-3
W_METRIC = 0.05
W_ARC    = 0.0   # no sphere.reg ground-truth needed for unsupervised variant

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def surfaces_to_verts(out: dict, hemi: str) -> torch.Tensor:
    """Extract (B, 3, N) from TopoRegNative output {"sphere.reg": {hemi: Surface}}."""
    return out["sphere.reg"][hemi].vertices.transpose(1, 2)  # (B, N, 3) -> (B, 3, N)


def expand_template(t: torch.Tensor, B: int, device: torch.device) -> torch.Tensor:
    """Expand (1, N) template sulc to (B, 1, N) on device."""
    return t.to(device).unsqueeze(0).expand(B, -1, -1)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: TopoRegNative,
    loader: DataLoader,
    criterion: RegistrationLossFull,
    template_sulc: dict,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """Run one training or validation epoch.  Pass optimizer=None for eval."""
    is_train = optimizer is not None
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    sums: dict[str, float] = {}

    with ctx:
        for batch in loader:
            sulc_subj   = batch["sulc_subj"][HEMI].to(DEVICE)    # (B, 1, 245762)
            sphere_reg  = batch["sphere_reg"][HEMI].to(DEVICE)   # (B, 3, 245762)
            B           = sulc_subj.shape[0]
            sulc_temp   = expand_template(template_sulc[HEMI], B, DEVICE)

            out = model(
                sulc_subj={HEMI: sulc_subj},
                sulc_temp={HEMI: sulc_temp},
            )

            pred_verts = surfaces_to_verts(out, HEMI)            # (B, 3, 245762)
            topology   = model.out_topology[HEMI]                # native topology

            loss_dict = criterion(
                pred_vertices=pred_verts,
                target_vertices=sphere_reg,
                pred_sulc=sulc_subj,
                target_sulc=sulc_temp,
                topology=topology,
            )

            if is_train:
                optimizer.zero_grad()
                loss_dict["total"].backward()
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

    # ------------------------------------------------------------------
    # Template sulcal depth at native resolution (no resampling)
    # ------------------------------------------------------------------
    print("Computing template sulcal depth at native resolution ...")
    template_sulc = compute_template_sulc(
        root_dir=DATA_ROOT,
        datasets=DATASETS,
        hemispheres=(HEMI,),
        ico_order=None,   # native resolution — no resampling
    )
    print(f"  template_sulc shape: {template_sulc[HEMI].shape}")  # (1, 245762)

    # ------------------------------------------------------------------
    # Dataset — native resolution (ico_order=None)
    # ------------------------------------------------------------------
    dataset = SphereRegDataset(
        root_dir=DATA_ROOT,
        datasets=DATASETS,
        hemispheres=(HEMI,),
        ico_order=None,   # native resolution — no KDTree resampling
        template_sulc=template_sulc,
    )
    print(dataset)

    n_val   = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("Building TopoRegNative ...")
    model = TopoRegNative(
        in_order=IN_ORDER,
        in_channels=1,
        encoder_channels=32,
        reg_channels=dict(encoder=[64, 64, 64, 64], decoder=[64, 64, 64]),
        n_steps=1,
        sphere_radius=100.0,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TopoRegNative  in_order={IN_ORDER}  params={n_params:,}")
    print(f"Active ico scales: {model.active_topologies}  + native")

    # ------------------------------------------------------------------
    # Optimizer & loss
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    criterion = RegistrationLossFull(
        w_arc=W_ARC,
        w_sulc=W_SULC,
        w_area=W_AREA,
        w_metric=W_METRIC,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(1, N_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, criterion, template_sulc, optimizer)

        if epoch % 5 == 0 or epoch == 1:
            val_metrics = run_epoch(model, val_loader, criterion, template_sulc, None)
            t_str = "  ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
            v_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            print(f"Epoch {epoch:3d}  train: {t_str}")
            print(f"          val:   {v_str}")
        else:
            print(f"Epoch {epoch:3d}  train_total={train_metrics['total']:.4f}")


if __name__ == "__main__":
    main()
