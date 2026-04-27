import torch
import torch.nn as nn

from brainnet.mesh.surface import Surface
from brainnet.modules.losses_surface import (
    SphericalArcLoss,
    OrientedAreaLoss,
    MetricDistortionLoss,
)


class RegistrationLossFull(nn.Module):
    """
    Full registration loss for spherical registration.

    total =
        w_arc    * SphericalArcLoss
      + w_sulc   * MSE(pred_sulc, target_sulc)
      + w_area   * OrientedAreaLoss
      + w_metric * MetricDistortionLoss

    Expected inputs
    ---------------
    pred_vertices   : [B, 3, M]
    target_vertices : [B, 3, M]
    pred_sulc       : [B, 1, M]
    target_sulc     : [B, 1, M]
    topology        : topology object from the repo
    """

    def __init__(
        self,
        w_arc: float = 1.0,
        w_sulc: float = 1.0,
        w_area: float = 1.0,
        w_metric: float = 1.0,
        radius: float | None = None,
    ):
        super().__init__()

        self.w_arc = w_arc
        self.w_sulc = w_sulc
        self.w_area = w_area
        self.w_metric = w_metric

        self.arc_loss = SphericalArcLoss(radius=radius)
        self.mse = nn.MSELoss()
        self.area_loss = OrientedAreaLoss(key="original_area")
        self.metric_loss = MetricDistortionLoss(key="original_edge_norm")

    @staticmethod
    def _check_shapes(
        pred_vertices: torch.Tensor,
        target_vertices: torch.Tensor,
        pred_sulc: torch.Tensor,
        target_sulc: torch.Tensor,
    ):
        if pred_vertices.ndim != 3 or pred_vertices.shape[1] != 3:
            raise ValueError(
                f"pred_vertices must have shape [B, 3, M], got {tuple(pred_vertices.shape)}"
            )
        if target_vertices.ndim != 3 or target_vertices.shape[1] != 3:
            raise ValueError(
                f"target_vertices must have shape [B, 3, M], got {tuple(target_vertices.shape)}"
            )
        if pred_sulc.ndim != 3 or pred_sulc.shape[1] != 1:
            raise ValueError(
                f"pred_sulc must have shape [B, 1, M], got {tuple(pred_sulc.shape)}"
            )
        if target_sulc.ndim != 3 or target_sulc.shape[1] != 1:
            raise ValueError(
                f"target_sulc must have shape [B, 1, M], got {tuple(target_sulc.shape)}"
            )

        if pred_vertices.shape[0] != target_vertices.shape[0]:
            raise ValueError("Batch size mismatch between pred_vertices and target_vertices.")
        if pred_sulc.shape[0] != target_sulc.shape[0]:
            raise ValueError("Batch size mismatch between pred_sulc and target_sulc.")
        if pred_vertices.shape[0] != pred_sulc.shape[0]:
            raise ValueError("Batch size mismatch between vertices and sulc.")

        if pred_vertices.shape[2] != target_vertices.shape[2]:
            raise ValueError("Number of vertices mismatch between pred_vertices and target_vertices.")
        if pred_sulc.shape[2] != target_sulc.shape[2]:
            raise ValueError("Number of vertices mismatch between pred_sulc and target_sulc.")
        if pred_vertices.shape[2] != pred_sulc.shape[2]:
            raise ValueError("Number of vertices mismatch between vertices and sulc.")

    @staticmethod
    def _attach_reference_geometry(pred_surface: Surface, ref_surface: Surface) -> Surface:
        """
        Store reference geometry on pred_surface for regularization losses.
        """
        pred_surface.face_data["original_area"] = ref_surface.compute_face_areas().detach()
        pred_surface.face_data["original_edge_norm"] = ref_surface.compute_edge_norm().detach()
        return pred_surface

    def forward(
        self,
        pred_vertices: torch.Tensor,
        target_vertices: torch.Tensor,
        pred_sulc: torch.Tensor,
        target_sulc: torch.Tensor,
        topology,
    ):
        self._check_shapes(pred_vertices, target_vertices, pred_sulc, target_sulc)

        pred_surface = Surface(pred_vertices.transpose(1, 2), topology)
        target_surface = Surface(target_vertices.transpose(1, 2), topology)

        pred_surface = self._attach_reference_geometry(pred_surface, target_surface)

        loss_arc = self.arc_loss(pred_surface, target_surface)
        loss_sulc = self.mse(pred_sulc, target_sulc)
        loss_area = self.area_loss(pred_surface)
        loss_metric = self.metric_loss(pred_surface)

        total = (
            self.w_arc * loss_arc
            + self.w_sulc * loss_sulc
            + self.w_area * loss_area
            + self.w_metric * loss_metric
        )

        return {
            "total": total,
            "arc": loss_arc,
            "sulc": loss_sulc,
            "area": loss_area,
            "metric": loss_metric,
        }