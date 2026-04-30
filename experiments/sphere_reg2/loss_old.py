"""
Loss minimale: NCC (Normalized Cross-Correlation) tra:
  - sulc_subj sulla mesh del soggetto
  - sulc_temp campionata dalla griglia 2D alle posizioni predette

NCC = -1 (peggio) ... +1 (perfetto). Restituiamo -NCC per minimizzare.
"""

import torch

from experiments.sphere_reg2.dataset import sample_grid_at_vertices


def ncc_loss(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Negative normalized cross-correlation along the last dim.

    a, b : (B, 1, N) or (B, N)
    Returns scalar.
    """
    a = a.flatten(1)
    b = b.flatten(1)

    a_c = a - a.mean(dim=-1, keepdim=True)
    b_c = b - b.mean(dim=-1, keepdim=True)

    num = (a_c * b_c).sum(dim=-1)
    den = torch.sqrt((a_c ** 2).sum(dim=-1) * (b_c ** 2).sum(dim=-1) + eps)

    return -(num / den).mean()


class SulcalAlignmentLoss(torch.nn.Module):
    """Compute NCC between subject sulc and template sulc resampled at predicted vertices."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_verts: torch.Tensor,       # (B, 3, N) posizioni predette
        sulc_subj: torch.Tensor,        # (B, 1, N) sulc del soggetto
        template_grid: torch.Tensor,    # (B, 1, H, W) o (1, H, W) griglia template
    ) -> dict[str, torch.Tensor]:

        sulc_temp_at_pred = sample_grid_at_vertices(template_grid, pred_verts)  # (B, 1, N)

        loss_sulc = ncc_loss(sulc_subj, sulc_temp_at_pred)

        return dict(total=loss_sulc, sulc=loss_sulc)