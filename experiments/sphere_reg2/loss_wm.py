"""
Loss totale: sulc + metric distortion + folding penalty.

  - L_sulc:    MSE tra sulc_subj e template alle pred verts
  - L_metric:  MSE(edge_pred, edge_ref_wm * scale)
               riferimento = WM surface del soggetto
               scale globale per ignorare scaling uniforme
  - L_folding: penalizza i triangoli con normale rivolta verso l'interno
               (folded), in modo smooth.
"""

import torch
import torch.nn.functional as F

from experiments.sphere_reg2.dataset_wm import sample_grid_at_vertices


def compute_edge_lengths(verts: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Lunghezze degli edge.

    verts: (B, N, 3)
    edges: (E, 2)

    Returns: (B, E)
    """
    v0 = verts[:, edges[:, 0]]
    v1 = verts[:, edges[:, 1]]
    return (v1 - v0).norm(dim=-1)


def faces_to_edges(faces: torch.Tensor) -> torch.Tensor:
    """Estrae edge unici dalle facce."""
    e0 = faces[:, [0, 1]]
    e1 = faces[:, [1, 2]]
    e2 = faces[:, [2, 0]]
    edges = torch.cat([e0, e1, e2], dim=0)
    edges, _ = torch.sort(edges, dim=1)
    edges = torch.unique(edges, dim=0)
    return edges


def compute_folding_penalty(verts: torch.Tensor, faces: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Penalizza i triangoli con normale rivolta verso il centro (folded).

    Per ogni triangolo:
      - n = normale (cross product)
      - b = baricentro (centro del triangolo)
      - d = n . b (prodotto scalare)

    Triangolo orientato verso l'esterno (corretto): d > 0
    Triangolo invertito (folded):                     d < 0

    La penalty e' relu(margin - d).pow(2):
      - margin = 0:   penalizza solo i triangoli effettivamente folded
      - margin > 0:   penalizza anche i triangoli "quasi flippati"

    verts: (B, N, 3)
    faces: (F, 3)
    Returns: scalar (media sui triangoli)
    """
    v0 = verts[:, faces[:, 0]]
    v1 = verts[:, faces[:, 1]]
    v2 = verts[:, faces[:, 2]]

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    normals_unit = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)

    barycenters = (v0 + v1 + v2) / 3
    bary_unit = barycenters / (barycenters.norm(dim=-1, keepdim=True) + 1e-8)

    d = (normals_unit * bary_unit).sum(dim=-1)
    penalty = F.relu(margin - d).pow(2).mean()
    return penalty


class SulcalAlignmentLoss(torch.nn.Module):
    """Loss totale: sulc + metric (vs WM) + folding penalty."""

    def __init__(
        self,
        w_sulc: float = 1.0,
        w_metric: float = 0.1,
        w_folding: float = 0.0,
        folding_margin: float = 0.1,
    ):
        """
        Args:
            w_sulc:        peso del termine di allineamento sulc
            w_metric:      peso della metric distortion (edge lengths vs WM)
            w_folding:     peso del folding penalty (consigliato 10-100)
            folding_margin: 0 = penalizza solo flippati,
                            0.1 = penalizza anche i "quasi flip"
        """
        super().__init__()
        self.w_sulc = w_sulc
        self.w_metric = w_metric
        self.w_folding = w_folding
        self.folding_margin = folding_margin

        self._faces = None
        self._edges = None

    def _setup_topology(self, faces: torch.Tensor):
        if self._faces is None or self._faces.device != faces.device:
            self._faces = faces.long()
            self._edges = faces_to_edges(self._faces)

    def forward(
        self,
        pred_verts: torch.Tensor,             # (B, 3, N)
        sulc_subj: torch.Tensor,              # (B, 1, N)
        template_grid: torch.Tensor,          # (B, 1, H, W)
        topology=None,
        ref_verts_metric: torch.Tensor = None,# (B, 3, N) WM surface
        faces: torch.Tensor = None,
    ):
        # === Sulc loss ===
        sulc_temp_at_pred = sample_grid_at_vertices(template_grid, pred_verts)
        loss_sulc = F.mse_loss(sulc_subj, sulc_temp_at_pred)

        total = self.w_sulc * loss_sulc

        loss_metric = torch.zeros((), device=pred_verts.device)
        loss_folding = torch.zeros((), device=pred_verts.device)

        if self.w_metric > 0 or self.w_folding > 0:
            if faces is None and topology is not None:
                faces = topology.faces
            assert faces is not None, "faces o topology richieste"
            self._setup_topology(faces.to(pred_verts.device))

            # (B, 3, N) -> (B, N, 3)
            pred_v = pred_verts.transpose(1, 2)

            # === Metric loss (riferimento = WM surface, con scale globale) ===
            if self.w_metric > 0:
                assert ref_verts_metric is not None, "ref_verts_metric richiesta per metric loss"
                ref_v_metric = ref_verts_metric.transpose(1, 2).detach()
                edge_pred = compute_edge_lengths(pred_v, self._edges)
                edge_ref = compute_edge_lengths(ref_v_metric, self._edges)
                scale = edge_pred.sum() / (edge_ref.sum() + 1e-8)
                loss_metric = F.mse_loss(edge_pred, edge_ref * scale)
                total = total + self.w_metric * loss_metric

            # === Folding penalty (smooth) ===
            if self.w_folding > 0:
                loss_folding = compute_folding_penalty(
                    pred_v, self._faces, margin=self.folding_margin
                )
                total = total + self.w_folding * loss_folding

        return {
            "total": total,
            "sulc": loss_sulc,
            "metric": loss_metric,
            "folding": loss_folding,
        }