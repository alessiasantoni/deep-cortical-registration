import torch
import torch.nn.functional as F

from brainnet.mesh.surface import Surface
from brainnet.modules.losses_surface import OrientedAreaLoss
from experiments.sphere_reg2.dataset_wm import sample_grid_at_vertices


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


class SulcalAlignmentLoss(torch.nn.Module):
    """Loss totale: sulc + metric vs WM + oriented area loss."""

    def __init__(
        self,
        w_sulc: float = 1.0,
        w_metric: float = 0.1,
        w_area: float = 0.01,
    ):
        super().__init__()
        self.w_sulc = w_sulc
        self.w_metric = w_metric
        self.w_area = w_area

        self.area_loss = OrientedAreaLoss(key="original_area")

        self._faces = None
        self._edges = None

    def _setup_topology(self, faces: torch.Tensor):
        if self._faces is None or self._faces.device != faces.device:
            self._faces = faces.long()
            self._edges = faces_to_edges(self._faces)

    @staticmethod
    def _attach_reference_area(pred_surface, ref_surface):
        pred_surface.face_data["original_area"] = (
            ref_surface.compute_face_areas().detach()
        )

    def forward(
        self,
        pred_verts: torch.Tensor,              # (B, 3, N)
        sulc_subj: torch.Tensor,               # (B, 1, N)
        template_grid: torch.Tensor,           # (B, 1, H, W)
        topology=None,
        ref_verts_metric: torch.Tensor = None, # (B, 3, N) WM surface
        ref_verts_area: torch.Tensor = None,   # (B, 3, N) reference sphere/template
        faces: torch.Tensor = None,
    ):
        # === Sulc loss ===
        sulc_temp_at_pred = sample_grid_at_vertices(template_grid, pred_verts)
        loss_sulc = F.mse_loss(sulc_subj, sulc_temp_at_pred)

        total = self.w_sulc * loss_sulc

        loss_metric = torch.zeros((), device=pred_verts.device)
        loss_area = torch.zeros((), device=pred_verts.device)

        if self.w_metric > 0 or self.w_area > 0:
            if faces is None and topology is not None:
                faces = topology.faces

            assert faces is not None, "faces o topology richieste"
            self._setup_topology(faces.to(pred_verts.device))

            pred_v = pred_verts.transpose(1, 2)  # (B, N, 3)

            # === Metric loss: invariata ===
            if self.w_metric > 0:
                assert ref_verts_metric is not None, "ref_verts_metric richiesta per metric loss"

                ref_v_metric = ref_verts_metric.transpose(1, 2).detach()

                edge_pred = compute_edge_lengths(pred_v, self._edges)
                edge_ref = compute_edge_lengths(ref_v_metric, self._edges)

                scale = edge_pred.sum() / (edge_ref.sum() + 1e-8)

                loss_metric = F.mse_loss(edge_pred, edge_ref * scale)
                total = total + self.w_metric * loss_metric

            # === Oriented Area Loss al posto del folding penalty ===
            if self.w_area > 0:
                assert ref_verts_area is not None, "ref_verts_area richiesta per area loss"
                assert topology is not None, "topology richiesta per OrientedAreaLoss"

                ref_surface = Surface(
                    ref_verts_area.transpose(1, 2).detach(),
                    topology
                )

                pred_surface = Surface(
                    pred_verts.transpose(1, 2),
                    topology
                )

                self._attach_reference_area(pred_surface, ref_surface)

                loss_area = self.area_loss(pred_surface).mean()
                total = total + self.w_area * loss_area

        return {
            "total": total,
            "sulc": loss_sulc,
            "metric": loss_metric,
            "area": loss_area,
        }