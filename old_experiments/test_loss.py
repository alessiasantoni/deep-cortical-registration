import torch

from brainnet.mesh.surface import load_deepsurfer_template
from experiments.sphere_reg.sphere_reg_baseline import SphereRegBaseline
from experiments.sphere_reg.registration_loss import RegistrationLossFull


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    template = load_deepsurfer_template(5, "white", ["lh"])
    topology = template["lh"].topology

    model = SphereRegBaseline(
        topology=topology,
        hidden_channels=32,
        encoder_layers=2,
        deform_channels=(64, 64, 64, 64),
        step_size=0.1,
        shared_encoder=True,
    ).to(device)

    criterion = RegistrationLossFull(
        w_arc=1.0,
        w_sulc=0.5,
        w_area=0.1,
        w_metric=0.1,
    )

    subject_vertices = template["lh"].vertices.clone().to(device)

    if subject_vertices.ndim != 3:
        raise ValueError(f"Unexpected vertices shape: {tuple(subject_vertices.shape)}")

    if subject_vertices.shape[-1] == 3:
        subject_vertices = subject_vertices.transpose(1, 2)

    if subject_vertices.shape[1] != 3:
        raise ValueError(
            f"Could not reshape vertices to [B, 3, M], got {tuple(subject_vertices.shape)}"
        )

    B = subject_vertices.shape[0]
    M = subject_vertices.shape[-1]

    subject_sulc = torch.randn(B, 1, M, device=device)
    template_sulc = torch.randn(B, 1, M, device=device)

    # target finto per test
    target_vertices = subject_vertices.clone()

    with torch.no_grad():
        out = model(subject_vertices, subject_sulc, template_sulc)

    # Per ora uso il subject sulc come "pred_sulc"
    # Più avanti qui metterai il sulc warped davvero
    pred_sulc = subject_sulc

    loss_dict = criterion(
        pred_vertices=out["registered_vertices"],
        target_vertices=target_vertices,
        pred_sulc=pred_sulc,
        target_sulc=template_sulc,
        topology=topology,
    )

    for k, v in loss_dict.items():
        print(f"{k}: {v.item():.6f}")


if __name__ == "__main__":
    main()