"""
Loss functions for training the spherical registration network.

Combines a sulcal-depth (sulc) alignment term with mesh-quality terms that
penalize folded (flipped) triangles, computed from the signed area of each
face on the predicted sphere: a metric-preservation term against the WM
surface, a folded-area penalty, and a differentiable folded-triangle count.
SulcalAlignmentLoss combines these into a single weighted total loss.
"""

import torch
import torch.nn.functional as F

from experiments.sphere_reg2.dataset import sample_grid_at_vertices


def compute_signed_face_areas(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Signed area of each triangle, sphere-oriented.

    Args:
        verts: (B, N, 3) — vertices of the predicted sphere
        faces: (F, 3)    — triangle indices

    Returns:
        (B, F) — signed area. Positive if the triangle is oriented
                 like the sphere (normal points outward), negative
                 if the triangle is flipped.
    """
    v0 = verts[:, faces[:, 0]]                # (B, F, 3)
    v1 = verts[:, faces[:, 1]]
    v2 = verts[:, faces[:, 2]]

    # Geometric normal (magnitude = 2 * unsigned area)
    normal = torch.cross(v1 - v0, v2 - v0, dim=-1)   # (B, F, 3)

    # Outward radial direction (normalized barycenter)
    bary = (v0 + v1 + v2) / 3.0                       # (B, F, 3)
    bary_unit = bary / (bary.norm(dim=-1, keepdim=True) + 1e-8)

    # Projection: signed area = 0.5 * (normal . bary_unit)
    signed_area = 0.5 * (normal * bary_unit).sum(dim=-1)   # (B, F)
    return signed_area


def folded_area_loss(verts: torch.Tensor, faces: torch.Tensor,
                     reduction: str = "sum") -> torch.Tensor:
    """Loss equal to the sum/mean of the areas of the flipped triangles.

    Args:
        verts: (B, N, 3)
        faces: (F, 3)
        reduction: 'sum' (total flipped area)

    Returns:
        scalar — loss; zero if there are no flips.
    """
    signed_area = compute_signed_face_areas(verts, faces)   # (B, F)

    # ReLU(-signed_area) -> 0 if positive, |area| if negative
    folded_magnitude = F.relu(-signed_area)                  # (B, F)

    if reduction == "sum":
        # Sum per batch sample, then mean over the batch
        return folded_magnitude.sum(dim=1).mean()
    elif reduction == "mean":
        return folded_magnitude.mean()
    else:
        raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction}")


def compute_edge_lengths(verts: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    v0 = verts[:, edges[:, 0]]
    v1 = verts[:, edges[:, 1]]
    return (v1 - v0).norm(dim=-1)


def faces_to_edges(faces: torch.Tensor) -> torch.Tensor:
    e0 = faces[:, [0, 1]]
    e1 = faces[:, [1, 2]]
    e2 = faces[:, [2, 0]]
    edges = torch.cat([e0, e1, e2], dim=0)
    edges, _ = torch.sort(edges, dim=1)
    edges = torch.unique(edges, dim=0)
    return edges


def folded_count_smooth(verts: torch.Tensor, faces: torch.Tensor,
                        temperature: float = 0.5) -> torch.Tensor:
    """Approximately counts the number of flipped triangles (differentiable).

    Args:
        verts: (B, N, 3)
        faces: (F, 3)
        temperature: sigmoid smoothing. Smaller = closer to a
                     binary count, but "sharper" gradients.
                     Default 0.5 is a good compromise.

    Returns:
        scalar — sum per batch sample, averaged over the batch.
                 Approximates the number of flipped triangles.
    """
    signed_area = compute_signed_face_areas(verts, faces)   # (B, F)
    # Sigmoid of the negative area: 0 if signed_area >> 0, ~1 if signed_area << 0
    folded_indicator = torch.sigmoid(-signed_area / temperature)
    return folded_indicator.sum(dim=1).mean()


class SulcalAlignmentLoss(torch.nn.Module):
    """Total loss: sulc + metric vs WM + area (folded) + count (folded smooth)."""

    def __init__(
        self,
        w_sulc: float = 1.0,
        w_metric: float = 0.1,
        w_area: float = 0.01,
        w_count: float = 0.0,
        count_temperature: float = 0.5,  # Sigmoid temperature for the count term
        folded_reduction: str = "sum",
    ):
        super().__init__()
        self.w_sulc = w_sulc
        self.w_metric = w_metric
        self.w_area = w_area
        self.w_count = w_count
        self.count_temperature = count_temperature
        self.folded_reduction = folded_reduction

        self._faces = None
        self._edges = None

    def _setup_topology(self, faces: torch.Tensor):
        if self._faces is None or self._faces.device != faces.device:
            self._faces = faces.long()
            self._edges = faces_to_edges(self._faces)

    def forward(
        self,
        pred_verts: torch.Tensor,              # (B, 3, N)
        sulc_subj: torch.Tensor,               # (B, 1, N)
        template_grid: torch.Tensor,           # (B, 1, H, W)
        topology=None,
        ref_verts_metric: torch.Tensor = None, # (B, 3, N) WM surface
        faces: torch.Tensor = None,
    ):
        # === Sulc loss ===
        sulc_temp_at_pred = sample_grid_at_vertices(template_grid, pred_verts)
        loss_sulc = F.mse_loss(sulc_subj, sulc_temp_at_pred)

        total = self.w_sulc * loss_sulc

        loss_metric = torch.zeros((), device=pred_verts.device)
        loss_area = torch.zeros((), device=pred_verts.device)
        loss_count = torch.zeros((), device=pred_verts.device)

        if self.w_metric > 0 or self.w_area > 0 or self.w_count > 0:
            if faces is None and topology is not None:
                faces = topology.faces

            assert faces is not None, "faces or topology required"
            self._setup_topology(faces.to(pred_verts.device))

            pred_v = pred_verts.transpose(1, 2)  # (B, N, 3)

            # === Metric loss ===
            if self.w_metric > 0:
                assert ref_verts_metric is not None, "ref_verts_metric required for metric loss"

                ref_v_metric = ref_verts_metric.transpose(1, 2).detach()

                edge_pred = compute_edge_lengths(pred_v, self._edges)
                edge_ref = compute_edge_lengths(ref_v_metric, self._edges)

                scale = edge_pred.sum() / (edge_ref.sum() + 1e-8)

                loss_metric = F.mse_loss(edge_pred, edge_ref * scale)
                total = total + self.w_metric * loss_metric

            # === Area loss ===
            if self.w_area > 0:
                loss_area = folded_area_loss(
                   pred_v, self._faces,
                   reduction=self.folded_reduction,
                )
                total = total + self.w_area * loss_area

            # === Count loss ===
            if self.w_count > 0:
                loss_count = folded_count_smooth(
                    pred_v, self._faces, temperature=self.count_temperature,
                )
                total = total + self.w_count * loss_count

        return {
            "total": total,
            "sulc": loss_sulc,
            "metric": loss_metric,
            "area": loss_area,
            "count": loss_count,
        }