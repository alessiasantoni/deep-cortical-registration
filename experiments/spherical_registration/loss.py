"""
Losses for the spherical registration network: sulc alignment with the template,
plus terms that keep the predicted mesh regular (edge lengths close to the WM
surface, no flipped triangles).
"""

import torch
import torch.nn.functional as F

from experiments.spherical_registration.dataset import sample_grid_at_vertices


def compute_signed_face_areas(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """verts (B, N, 3), faces (F, 3) -> (B, F) area of each triangle, negative if flipped."""
    v0 = verts[:, faces[:, 0]]
    v1 = verts[:, faces[:, 1]]
    v2 = verts[:, faces[:, 2]]

    normal = torch.cross(v1 - v0, v2 - v0, dim=-1)   # its norm is twice the area
    # on a sphere the outward direction of a face is just its (normalized) barycenter
    bary = (v0 + v1 + v2) / 3.0
    bary_unit = bary / (bary.norm(dim=-1, keepdim=True) + 1e-8)

    signed_area = 0.5 * (normal * bary_unit).sum(dim=-1)
    return signed_area


def folded_area_loss(verts: torch.Tensor, faces: torch.Tensor,
                     reduction: str = "sum") -> torch.Tensor:
    """Total (or mean) area of the flipped triangles, 0 when nothing is flipped."""
    signed_area = compute_signed_face_areas(verts, faces)
    folded_magnitude = F.relu(-signed_area)

    if reduction == "sum":
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
    """Soft count of flipped triangles (a sigmoid instead of a step, so it has gradients).
    Lower temperature = closer to the real count but sharper gradients."""
    signed_area = compute_signed_face_areas(verts, faces)
    folded_indicator = torch.sigmoid(-signed_area / temperature)
    return folded_indicator.sum(dim=1).mean()


class SulcalAlignmentLoss(torch.nn.Module):
    """sulc + metric (edge lengths vs WM) + area (flipped area) + count (soft flipped count)."""

    def __init__(
        self,
        w_sulc: float = 1.0,
        w_metric: float = 0.1,
        w_area: float = 0.01,
        w_count: float = 0.0,
        count_temperature: float = 0.5,
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
        # edges are computed once and reused, the mesh never changes
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

            pred_v = pred_verts.transpose(1, 2)

            if self.w_metric > 0:
                assert ref_verts_metric is not None, "ref_verts_metric required for metric loss"

                ref_v_metric = ref_verts_metric.transpose(1, 2).detach()

                edge_pred = compute_edge_lengths(pred_v, self._edges)
                edge_ref = compute_edge_lengths(ref_v_metric, self._edges)

                # sphere and WM surface have different sizes: compare the edge lengths
                # after rescaling the WM ones to the same total length
                scale = edge_pred.sum() / (edge_ref.sum() + 1e-8)

                loss_metric = F.mse_loss(edge_pred, edge_ref * scale)
                total = total + self.w_metric * loss_metric

            if self.w_area > 0:
                loss_area = folded_area_loss(
                   pred_v, self._faces,
                   reduction=self.folded_reduction,
                )
                total = total + self.w_area * loss_area

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
