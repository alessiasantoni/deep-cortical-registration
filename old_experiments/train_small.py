import torch
from torch.utils.data import DataLoader
from brainnet.modules.graph import GenericSurfaceModule
from experiments.sphere_reg.dataset import SphereRegDataset, compute_template_sulc
from experiments.sphere_reg.sphere_reg_baseline import SphereRegBaseline
from experiments.sphere_reg.registration_loss import RegistrationLossFull

DATA_ROOT = "/projects/brainnet-data/sphere-reg"
ICO_ORDER = 5
HEMI = "lh"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_EPOCHS = 5
LR = 1e-3

def main():
    print(f"Device: {DEVICE}")

    # Template
    print("Computing template sulcal depth...")
    template_sulc = compute_template_sulc(
        root_dir=DATA_ROOT,
        datasets=["Buckner40"],
        hemispheres=(HEMI,),
        ico_order=ICO_ORDER,
        max_subjects=10,  # ← piccolo!
    )

    # Dataset con pochi soggetti
    dataset = SphereRegDataset(
        root_dir=DATA_ROOT,
        datasets=["Buckner40"],  # ← un solo dataset
        hemispheres=(HEMI,),
        ico_order=ICO_ORDER,
        template_sulc=template_sulc,
    )
    print(f"Dataset: {len(dataset)} subjects")

    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    # Modello
    #gsm = GenericSurfaceModule(0, ICO_ORDER, ICO_ORDER, topology="FsAverageTopology")
    gsm = GenericSurfaceModule(0, ICO_ORDER, ICO_ORDER, topology="DeepSurferTopology")
    topology = gsm.topologies[ICO_ORDER]

    model = SphereRegBaseline(
        topology=topology,
        hidden_channels=32,
        encoder_layers=2,
        deform_channels=(64, 64, 64, 64),
        step_size=0.1,
    ).to(DEVICE)

    # Loss e optimizer
    criterion = RegistrationLossFull(
        w_arc=0.0,
        w_sulc=1.0,
        w_area=0.1,
        w_metric=0.1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in loader:
            sulc_subj  = batch["sulc_subj"][HEMI].to(DEVICE)
            ico_verts  = batch["ico_verts"][HEMI].to(DEVICE)
            sulc_temp  = batch["template_sulc"][HEMI].to(DEVICE)
            sphere_reg = batch["sphere_reg"][HEMI].to(DEVICE)

            optimizer.zero_grad()
            out = model(ico_verts, sulc_subj, sulc_temp)

            loss_dict = criterion(
                pred_vertices=out["registered_vertices"],
                target_vertices=sphere_reg,
                pred_sulc=sulc_subj,
                target_sulc=sulc_temp,
                topology=topology,
            )
            loss = loss_dict["total"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{N_EPOCHS} - Loss: {total_loss/len(loader):.6f}")

    print("Training done!")
    torch.save(model.state_dict(), "sphere_reg_baseline.pth")

if __name__ == "__main__":
    main()