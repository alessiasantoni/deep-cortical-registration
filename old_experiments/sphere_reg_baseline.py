import torch
import torch.nn as nn

from brainnet.modules.graph.layers import (
    GraphConvolutionBlock,
    ResidualGraphConvolutionDeformationBlock,
)


class SimpleGraphEncoder(nn.Module):
    def __init__(self, topology, in_channels=1, hidden_channels=32, n_layers=2):
        super().__init__()
        layers = []
        cur_in = in_channels
        for _ in range(n_layers):
            layers.append(
                GraphConvolutionBlock(
                    cur_in,
                    hidden_channels,
                    topology.conv_index_reduce,
                    topology.conv_index_gather,
                )
            )
            cur_in = hidden_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SphereRegBaseline(nn.Module):
    def __init__(
        self,
        topology,
        hidden_channels: int = 32,
        encoder_layers: int = 2,
        deform_channels=(64, 64, 64, 64),
        step_size: float = 1.0,
        shared_encoder: bool = True,
    ):
        super().__init__()
        self.topology = topology
        self.step_size = step_size

        self.subject_encoder = SimpleGraphEncoder(
            topology=topology,
            in_channels=1,
            hidden_channels=hidden_channels,
            n_layers=encoder_layers,
        )

        if shared_encoder:
            self.template_encoder = self.subject_encoder
        else:
            self.template_encoder = SimpleGraphEncoder(
                topology=topology,
                in_channels=1,
                hidden_channels=hidden_channels,
                n_layers=encoder_layers,
            )

        match_channels = hidden_channels * 4

        self.deform_head = ResidualGraphConvolutionDeformationBlock(
            in_channels=match_channels,
            graph_channels=list(deform_channels),
            out_channels=3,
            reduce_index=topology.conv_index_reduce,
            gather_index=topology.conv_index_gather,
            n_residual_blocks=3,
            out_init_values=0.0,
        )

    @staticmethod
    def normalize_to_sphere(vertices: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norms = torch.linalg.norm(vertices, dim=1, keepdim=True).clamp_min(eps)
        return vertices / norms

    @staticmethod
    def build_matching_features(
        feat_subject: torch.Tensor,
        feat_template: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                feat_subject,
                feat_template,
                feat_subject - feat_template,
                feat_subject * feat_template,
            ],
            dim=1,
        )

    def forward(
        self,
        subject_vertices: torch.Tensor,
        subject_sulc: torch.Tensor,
        template_sulc: torch.Tensor,
    ):
        if subject_vertices.ndim != 3 or subject_vertices.shape[1] != 3:
            raise ValueError(
                f"subject_vertices must have shape [B, 3, M], got {tuple(subject_vertices.shape)}"
            )

        if subject_sulc.ndim != 3 or subject_sulc.shape[1] != 1:
            raise ValueError(
                f"subject_sulc must have shape [B, 1, M], got {tuple(subject_sulc.shape)}"
            )

        if template_sulc.ndim != 3 or template_sulc.shape[1] != 1:
            raise ValueError(
                f"template_sulc must have shape [B, 1, M], got {tuple(template_sulc.shape)}"
            )

        if not (
            subject_vertices.shape[0] == subject_sulc.shape[0] == template_sulc.shape[0]
        ):
            raise ValueError("Batch size mismatch between inputs.")

        if not (
            subject_vertices.shape[2] == subject_sulc.shape[2] == template_sulc.shape[2]
        ):
            raise ValueError("Number of vertices M mismatch between inputs.")

        feat_subject = self.subject_encoder(subject_sulc)
        feat_template = self.template_encoder(template_sulc)

        match = self.build_matching_features(feat_subject, feat_template)
        delta = self.deform_head(match)

        updated_vertices = subject_vertices + self.step_size * delta
        registered_vertices = self.normalize_to_sphere(updated_vertices)

        return {
            "registered_vertices": registered_vertices,
            "delta": delta,
            "feat_subject": feat_subject,
            "feat_template": feat_template,
            "match": match,
        }