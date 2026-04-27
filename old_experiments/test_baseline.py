import torch

from brainnet.mesh.surface import load_deepsurfer_template
from experiments.sphere_reg.sphere_reg_baseline import SphereRegBaseline


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Carica una topologia vera del prof
    # Prova ordine 5 o 6; se non va, cambia con un ordine disponibile nel repo
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

    B = 1
    M = topology.n_vertices

    # Vertici iniziali: prendiamo una sfera vera e la usiamo come input
    subject_vertices = template["lh"].vertices.clone().to(device)  # [1, M, 3] oppure [1, 3, M] dipende dal repo

    # Adattiamo la shape a [B, 3, M]
    if subject_vertices.ndim != 3:
        raise ValueError(f"Unexpected vertices shape: {tuple(subject_vertices.shape)}")

    # Possibili casi: [B, M, 3] oppure [B, 3, M]
    if subject_vertices.shape[-1] == 3:
        subject_vertices = subject_vertices.transpose(1, 2)

    if subject_vertices.shape[1] != 3:
        raise ValueError(f"Could not reshape vertices to [B, 3, M], got {tuple(subject_vertices.shape)}")

    # Sulc finto per adesso
    subject_sulc = torch.randn(B, 1, M, device=device)
    template_sulc = torch.randn(B, 1, M, device=device)

    with torch.no_grad():
        out = model(subject_vertices, subject_sulc, template_sulc)

    print("subject_vertices      :", tuple(subject_vertices.shape))
    print("subject_sulc          :", tuple(subject_sulc.shape))
    print("template_sulc         :", tuple(template_sulc.shape))
    print("registered_vertices   :", tuple(out["registered_vertices"].shape))
    print("delta                 :", tuple(out["delta"].shape))
    print("feat_subject          :", tuple(out["feat_subject"].shape))
    print("feat_template         :", tuple(out["feat_template"].shape))
    print("match                 :", tuple(out["match"].shape))

    radii = torch.linalg.norm(out["registered_vertices"], dim=1)
    print("mean radius           :", radii.mean().item())
    print("min radius            :", radii.min().item())
    print("max radius            :", radii.max().item())


if __name__ == "__main__":
    main()