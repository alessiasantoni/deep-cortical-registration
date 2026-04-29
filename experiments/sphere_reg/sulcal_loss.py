
import torch
import torch.nn.functional as F
from brainnet.mesh.surface import Surface
from brainnet.modules.losses_surface import OrientedAreaLoss, MetricDistortionLoss
from experiments.sphere_reg.dataset import sample_spherical_grid


class SulcalAlignmentLoss(torch.nn.Module):
    """Loss using subject mesh and template grid."""

    def __init__(
        self,
        w_sulc: float = 1.0,
        w_area: float = 0.05,
        w_metric: float = 0.05,
    ):
        super().__init__()
        self.w_sulc   = w_sulc
        self.w_area   = w_area
        self.w_metric = w_metric

        self.area_loss   = OrientedAreaLoss(key="original_area")
        self.metric_loss = MetricDistortionLoss(key="original_edge_norm")

    @staticmethod
    def _attach_reference_geometry(pred: Surface, ref: Surface) -> None:
        pred.face_data["original_area"]      = ref.compute_face_areas().detach()
        pred.face_data["original_edge_norm"] = ref.compute_edge_norm().detach()

    def forward(
        self,
        pred_verts: torch.Tensor,       # (B, 3, N)
        sulc_subj: torch.Tensor,        # (B, 1, N)  mesh
        sulc_temp_grid: torch.Tensor,   # (B, 1, H, W)  griglia 2D
        topology,
        ref_verts: torch.Tensor,        # (B, 3, N) o (3, N)
    ) -> dict[str, torch.Tensor]:

     
        # Sampling of sulc_temp_grid on predicted positions
        sulc_temp_at_pred = sample_spherical_grid(sulc_temp_grid, pred_verts)

        # Comparison: subject value at vertex i vs template value at predicted position
        loss_sulc = F.mse_loss(sulc_subj, sulc_temp_at_pred)

        # Geometric regularization
        if ref_verts.ndim == 2:
            ref_verts = ref_verts.unsqueeze(0).expand(pred_verts.shape[0], -1, -1)
        ref_surface = Surface(ref_verts.transpose(1, 2).detach(), topology)
        pred_surface = Surface(pred_verts.transpose(1, 2), topology)
        self._attach_reference_geometry(pred_surface, ref_surface)

        loss_area   = self.area_loss(pred_surface)
        loss_metric = self.metric_loss(pred_surface)

        total = (
            self.w_sulc   * loss_sulc
            + self.w_area   * loss_area
            + self.w_metric * loss_metric
        )
        return dict(total=total, sulc=loss_sulc, area=loss_area, metric=loss_metric)
