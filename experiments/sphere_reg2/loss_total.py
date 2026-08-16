import torch
import torch.nn.functional as F

from brainnet.mesh.surface import Surface
from brainnet.modules.losses_surface import OrientedAreaLoss
from experiments.sphere_reg2.dataset import sample_grid_at_vertices

def compute_signed_face_areas(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area con segno di ogni triangolo, sfera-orientata.
    
    Args:
        verts: (B, N, 3) — vertici della sfera predetta
        faces: (F, 3)    — indici dei triangoli
    
    Returns:
        (B, F) — area con segno. Positiva se il triangolo è orientato
                 come la sfera (normale punta verso l'esterno), negativa
                 se il triangolo è flippato.
    """
    v0 = verts[:, faces[:, 0]]                # (B, F, 3)
    v1 = verts[:, faces[:, 1]]
    v2 = verts[:, faces[:, 2]]
    
    # Normale geometrica (modulo = 2 * area non segnata)
    normal = torch.cross(v1 - v0, v2 - v0, dim=-1)   # (B, F, 3)
    
    # Direzione radiale uscente (baricentro normalizzato)
    bary = (v0 + v1 + v2) / 3.0                       # (B, F, 3)
    bary_unit = bary / (bary.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Proiezione: area con segno = 0.5 * (normal . bary_unit)
    signed_area = 0.5 * (normal * bary_unit).sum(dim=-1)   # (B, F)
    return signed_area


def folded_area_loss(verts: torch.Tensor, faces: torch.Tensor,
                     reduction: str = "sum") -> torch.Tensor:
    """Loss pari alla somma/media delle aree dei triangoli flippati.
    
    Args:
        verts: (B, N, 3)
        faces: (F, 3)
        reduction: 'sum' (totale area flippata) 
    
    Returns:
        scalar — loss; zero se non ci sono flip.
    """
    signed_area = compute_signed_face_areas(verts, faces)   # (B, F)
    
    # ReLU(-signed_area) -> 0 se positivo, |area| se negativo
    folded_magnitude = F.relu(-signed_area)                  # (B, F)
    
    if reduction == "sum":
        # Somma per batch sample, poi media sul batch
        return folded_magnitude.sum(dim=1).mean()
    elif reduction == "mean":
        return folded_magnitude.mean()
    else:
        raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction}")



def compute_edge_lengths(verts: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    v0 = verts[:, edges[:, 0]]
    v1 = verts[:, edges[:, 1]]
    return (v1 - v0).norm(dim=-1)

def displacement_smoothness_loss(
    pred_verts: torch.Tensor,   # (B, N, 3)
    ref_verts: torch.Tensor,    # (B, N, 3) — sfera di input/riferimento
    edges: torch.Tensor,        # (E, 2)
) -> torch.Tensor:
    """Penalizza la rugosita' del campo di deformazione.

    Displacement d = pred - ref. Penalizza la differenza di displacement
    tra vertici adiacenti:
        L = mean_over_edges || d_i - d_j ||^2
    Vertici vicini che si muovono in modo simile -> deformazione smooth.
    Vertici vicini che divergono -> alta frequenza, penalizzata.
    """
    disp = pred_verts - ref_verts          # (B, N, 3)
    d0 = disp[:, edges[:, 0]]              # (B, E, 3)
    d1 = disp[:, edges[:, 1]]             # (B, E, 3)
    diff = d0 - d1                         # (B, E, 3)
    return (diff ** 2).sum(dim=-1).mean()  # scalar


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
    """Conta approssimativamente il numero di triangoli flippati (differenziabile).
    
    Args:
        verts: (B, N, 3)
        faces: (F, 3)
        temperature: smussamento della sigmoide. Piu' piccolo = piu' simile a
                     un conteggio binario, ma gradienti piu' "sharp".
                     Default 0.5 e' un buon compromesso.
    
    Returns:
        scalar — somma per batch sample, mediata sul batch.
                 Approssima il numero di triangoli flippati.
    """
    signed_area = compute_signed_face_areas(verts, faces)   # (B, F)
    # Sigmoide della negative area: 0 se signed_area >> 0, ~1 se signed_area << 0
    folded_indicator = torch.sigmoid(-signed_area / temperature)
    return folded_indicator.sum(dim=1).mean()


def folding_penalty(verts: torch.Tensor, faces: torch.Tensor,
                    w_area_rel: float = 1.0,
                    w_count_rel: float = 0.001,
                    temperature: float = 0.5,
                    reduction: str = "sum") -> dict:
    """Combined penalty anti-folding
    Returns a dict with:
        - "fold": (w_area_rel*area + w_count_rel*count)
        - "area_raw"
        - "count_raw"
    """
    signed_area = compute_signed_face_areas(verts, faces)   # (B, F)

    # Componente AREA: gravita' dei flip
    area_term = F.relu(-signed_area)                          # (B, F)
    if reduction == "sum":
        area_raw = area_term.sum(dim=1).mean()
    else:
        area_raw = area_term.mean()

    # Componente COUNT: numero di flip (smooth)
    count_term = torch.sigmoid(-signed_area / temperature)   # (B, F)
    count_raw = count_term.sum(dim=1).mean()

    # Termine combinato
    fold = w_area_rel * area_raw + w_count_rel * count_raw

    return {
        "fold": fold,
        "area_raw": area_raw,
        "count_raw": count_raw,
    }

class SulcalAlignmentLoss(torch.nn.Module):
    """Loss totale: sulc + metric vs WM + oriented area loss."""

    def __init__(
        self,
        w_sulc: float = 1.0,
        w_metric: float = 0.1,
        #w_fold = 1.0,
        #fold_area_rel: float = 1.0,     # Relative weight of area term
        #fold_count_rel: float = 0.001,  # Relative weight of count term
        w_area: float = 0.01,
        w_count: float = 0.0,
        w_smooth: float = 0.00,         # Weight for displacement smoothness loss      
        count_temperature: float = 0.5, # Sigmoid temperature for count term
        #area_loss_type: str = "folded",
        folded_reduction: str = "sum",
    ):
        super().__init__()
        self.w_sulc = w_sulc
        self.w_metric = w_metric
        #self.w_fold = w_fold
        self.count_temperature = count_temperature 
        self.folded_reduction = folded_reduction
        self.w_area = w_area
        self.w_count = w_count    
        self.w_smooth = w_smooth                     
        #self.area_loss_type = area_loss_type
        #self.area_loss = OrientedAreaLoss(key="original_area")
   
        #self.fold_area_rel = fold_area_rel
        #self.fold_count_rel = fold_count_rel

        #if area_loss_type == "oriented":
         #   self.area_loss = OrientedAreaLoss(key="original_area")
        #elif area_loss_type == "folded":
         #   self.area_loss = None  # useremo folded_area_loss direttamente
        #else:
         #   raise ValueError(f"area_loss_type sconosciuto: {area_loss_type}")
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
        loss_count = torch.zeros((), device=pred_verts.device)
        loss_smooth = torch.zeros((), device=pred_verts.device)

        if self.w_metric > 0 or self.w_area > 0 or self.w_count > 0 or self.w_smooth > 0: # NUOVO anche count e smooth
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
           
            # === Folding penalty combinata ===
            #if self.w_fold > 0:
            #    fold_dict = folding_penalty(
            #        pred_v, self._faces,
            #        w_area_rel=self.fold_area_rel,
            #        w_count_rel=self.fold_count_rel,
            #        temperature=self.count_temperature,
            #        reduction=self.folded_reduction,
            #    )

            #    loss_fold = fold_dict["fold"]
            #    fold_area_raw = fold_dict["area_raw"]
            #    fold_count_raw = fold_dict["count_raw"]
            #    total = total + self.w_fold * loss_fold

        # vecchia area loss che aveva area e count separati
            # === Oriented Area Loss al posto del folding penalty ===
            # === Area loss ===
            if self.w_area > 0:
                loss_area = folded_area_loss(
                   pred_v, self._faces,
                   reduction=self.folded_reduction,
                )
                total = total + self.w_area * loss_area
           
           # if self.w_area > 0:
            #    if self.area_loss_type == "folded":
             #           loss_area = folded_area_loss(
              #          pred_v, self._faces,
               #         reduction=self.folded_reduction,
                #    )
              #  elif self.area_loss_type == "oriented":
                    # Vecchia: OrientedAreaLoss di brainnet
               #     assert ref_verts_area is not None
               #     assert topology is not None
                #    ref_surface = Surface(ref_verts_area.transpose(1, 2).detach(), topology)
                 #   pred_surface = Surface(pred_verts.transpose(1, 2), topology)
                  #  self._attach_reference_area(pred_surface, ref_surface)
                   # loss_area = self.area_loss(pred_surface).mean()
                
              #  total = total + self.w_area * loss_area

            # === Count loss (NUOVO) ===           
            if self.w_count > 0:
                loss_count = folded_count_smooth(
                    pred_v, self._faces, temperature=self.count_temperature,
                )
                total = total + self.w_count * loss_count

            
            # === Smoothness del campo di deformazione (NUOVO) ===
            if self.w_smooth > 0:
                assert ref_verts_area is not None, "ref_verts_area richiesta per smoothness loss"
                ref_v_smooth = ref_verts_area.transpose(1, 2).detach()  # (B, N, 3)
                loss_smooth = displacement_smoothness_loss(
                    pred_v, ref_v_smooth, self._edges
                )
                total = total + self.w_smooth * loss_smooth
 

        return { 
            "total": total,
            "sulc": loss_sulc,
            "metric": loss_metric,
            #"fold": loss_fold,
            #"fold_area_raw": fold_area_raw,
            #"fold_count_raw": fold_count_raw,
            "area": loss_area,
            "count": loss_count,  
            "smooth": loss_smooth, # NUOVO
        }