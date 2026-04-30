import torch
import torch.nn.functional as F

from experiments.sphere_reg2.dataset import sample_grid_at_vertices


def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.flatten(1)
    return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + eps)


class SulcalAlignmentLoss(torch.nn.Module):
    """MSE between normalized subject sulc and template sulc resampled at predicted vertices."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_verts: torch.Tensor,       # (B, 3, N)
        sulc_subj: torch.Tensor,        # (B, 1, N)
        template_grid: torch.Tensor,    # (B, 1, H, W) or (1, H, W)
    ) -> dict[str, torch.Tensor]:

        sulc_temp_at_pred = sample_grid_at_vertices(template_grid, pred_verts)  # (B, 1, N)

        subj = zscore(sulc_subj)
        temp = zscore(sulc_temp_at_pred)

        loss_sulc = F.mse_loss(subj, temp)

        return dict(total=loss_sulc, sulc=loss_sulc)