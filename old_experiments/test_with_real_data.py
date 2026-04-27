"""End-to-end test: SphereRegBaseline + RegistrationLossFull with real HPC data.

Run:
    /home/s252125/miniforge3/envs/brainnet/bin/python \
        -m experiments.sphere_reg.test_with_real_data
"""

import torch
from torch.utils.data import DataLoader

from brainnet.modules.graph import GenericSurfaceModule

from experiments.sphere_reg.dataset import SphereRegDataset, compute_template_sulc
from experiments.sphere_reg.sphere_reg_baseline import SphereRegBaseline
from experiments.sphere_reg.registration_loss import RegistrationLossFull

DATA_ROOT = "/projects/brainnet-data/sphere-reg"
ICO_ORDER = 5       # 10 242 vertices
HEMI = "lh"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"Device: {DEVICE}")

    # ------------------------------------------------------------------
    # 1. Template sulcal depth (group mean from Buckner40)
    # ------------------------------------------------------------------
    print("Computing template sulcal depth from Buckner40 ...")
    template_sulc = compute_template_sulc(
        root_dir=DATA_ROOT,
        datasets=["Buckner40"],
        hemispheres=(HEMI,),
        ico_order=ICO_ORDER,
        max_subjects=10,
    )

    # ------------------------------------------------------------------
    # 2. Dataset & DataLoader
    # ------------------------------------------------------------------
    dataset = SphereRegDataset(
        root_dir=DATA_ROOT,
        datasets=["Buckner40"],
        hemispheres=(HEMI,),
        ico_order=ICO_ORDER,
        template_sulc=template_sulc,
    )
    print(dataset)

    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    sulc_subj   = batch["sulc_subj"][HEMI].to(DEVICE)    # (B, 1, N_ico)
    ico_verts   = batch["ico_verts"][HEMI].to(DEVICE)    # (B, 3, N_ico)
    sulc_temp   = batch["template_sulc"][HEMI].to(DEVICE)  # (B, 1, N_ico)
    sphere_reg  = batch["sphere_reg"][HEMI].to(DEVICE)   # (B, 3, N_ico)

    print(f"sulc_subj  : {tuple(sulc_subj.shape)}")
    print(f"ico_verts  : {tuple(ico_verts.shape)}")
    print(f"sulc_temp  : {tuple(sulc_temp.shape)}")
    print(f"sphere_reg : {tuple(sphere_reg.shape)}")

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    # Topology at the chosen ico order
    gsm = GenericSurfaceModule(0, ICO_ORDER, ICO_ORDER, topology="FsAverageTopology")
    topology = gsm.topologies[ICO_ORDER]

    model = SphereRegBaseline(
        topology=topology,
        hidden_channels=32,
        encoder_layers=2,
        deform_channels=(64, 64, 64, 64),
        step_size=0.1,
        shared_encoder=True,
    ).to(DEVICE)

    # ------------------------------------------------------------------
    # 4. Forward pass
    # ------------------------------------------------------------------
    with torch.no_grad():
        out = model(ico_verts, sulc_subj, sulc_temp)

    print(f"\nregistered_vertices: {tuple(out['registered_vertices'].shape)}")
    radii = torch.linalg.norm(out["registered_vertices"], dim=1)
    print(f"vertex radius  mean={radii.mean():.3f}  min={radii.min():.3f}  max={radii.max():.3f}")

    # ------------------------------------------------------------------
    # 5. Loss
    #    pred_vertices : deformed ico positions  (B, 3, N_ico)
    #    target_vertices: sphere.reg             (B, 3, N_ico)  -- ground truth
    #    pred_sulc: subject sulcal depth (used as proxy; ideally sampled at
    #               predicted positions, but that requires mesh interpolation)
    #    target_sulc: template sulcal depth
    # ------------------------------------------------------------------
    criterion = RegistrationLossFull(
        w_arc=0.0,    # disabled: arc loss needs sphere.reg as target, which is
                      # approx equal to ico_verts (circular) -- enable once you
                      # have a proper inverse mapping
        w_sulc=1.0,   # align subject sulcal depth with template
        w_area=0.1,   # regularize face areas
        w_metric=0.1, # regularize edge lengths
    )

    loss_dict = criterion(
        pred_vertices=out["registered_vertices"],
        target_vertices=sphere_reg,      # sphere.reg as registration target
        pred_sulc=sulc_subj,             # subject sulcal depth (input proxy)
        target_sulc=sulc_temp,           # template sulcal depth
        topology=topology,
    )

    print("\nLoss components:")
    for k, v in loss_dict.items():
        print(f"  {k:10s}: {v.item():.6f}")


if __name__ == "__main__":
    main()
