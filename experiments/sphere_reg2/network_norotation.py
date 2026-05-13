
"""Rete di registrazione sferica coarse-to-fine.

Per ogni scala in [ico_in, ..., ico_out]:
  Per ogni iterazione (n_iter):
    1. Sample template grid alle posizioni correnti
    2. Stack [sulc_subj, sulc_temp_at_r] come 2 canali
    3. UNet → deformazione 3D
    4. Update positions e project to sphere
  Subdivide alla scala successiva (se non l'ultima)"""


import torch
import torch.nn.functional as F

from brainnet.mesh.topology import DeepSurferTopology
from brainnet.modules.graph.layers import EdgeConvolutionBlock, EdgeConvolution
from brainnet.modules.graph.unet import UNet
from brainnet.mesh.surface import rotate

from experiments.sphere_reg2.dataset import sample_grid_at_vertices



class SphereRegNet(torch.nn.Module):
    """Coarse-to-fine spherical registration.

    Parameters
    ----------
    ico_in : int
        Ordine icosaedrale di partenza (più grossolano).
    ico_out : int
        Ordine icosaedrale finale (più fine).
    n_iter : int | list[int]
        Numero di iterazioni per ogni scala. Se intero, broadcast su tutte le scale.
    in_channels : int
        Canali in input (2 = sulc_subj + sulc_temp_resampled).
    base_channels : int
        Larghezza base dell'encoder/decoder UNet.
    sphere_radius : float
        Raggio della sfera.
    step_size : float
        Scala del campo di deformazione applicato ad ogni iterazione.
    """

    def __init__(
        self,
        ico_in: int = 2,
        ico_out: int = 6,
        n_iter: int | list[int] = 1,
        in_channels: int = 2,
        base_channels: int = 64,
        sphere_radius: float = 100.0,
        step_size: float = 0.1,
    ):
        super().__init__()
        assert ico_in <= ico_out, f"ico_in ({ico_in}) must be <= ico_out ({ico_out})"

        self.ico_in = ico_in
        self.ico_out = ico_out
        self.sphere_radius = sphere_radius
        self.step_size = step_size

        # Tutte le topologie da ico0 a ico_out
        self.topologies = DeepSurferTopology.recursive_subdivision(ico_out)

        # Scale attive (da ico_in a ico_out)
        self.active_orders = list(range(ico_in, ico_out + 1))

        # n_iter per ogni scala
        if isinstance(n_iter, int):
            n_iter = [n_iter] * len(self.active_orders)
        assert len(n_iter) == len(self.active_orders), \
            f"n_iter must have length {len(self.active_orders)}"
        self.n_iter = dict(zip(self.active_orders, n_iter))

        # Una UNet per ogni scala (con i parametri specifici)
        self.unets = torch.nn.ModuleDict()
        self.deforms = torch.nn.ModuleDict()

        for order in self.active_orders:
            topos = self.topologies[: order + 1]
            depth = min(4, order + 1)

            unet = UNet(
                in_channels=in_channels,
                topologies=topos,
                conv_module=EdgeConvolutionBlock,
                reduce="amax",
                channels=base_channels,
                max_depth=depth,
                n_conv=1,
            )
            self.unets[str(order)] = unet

            self.deforms[str(order)] = EdgeConvolution(
                in_channels=unet.out_ch,
                out_channels=3,
                reduce_index=topos[-1].conv_index_reduce,
                gather_index=topos[-1].conv_index_gather,
                bias=False,
                init_values=0.0,
            )
        print("Topology check:")
        for order in range(self.ico_out + 1):
            n = self.topologies[order].n_vertices
            print(f"  topologies[{order}] has {n} vertices")
        

    def project_to_sphere(self, r: torch.Tensor) -> torch.Tensor:
        """Project (B, 3, N) onto sphere, robust to vectors near zero."""
        norm = r.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return (r / norm) * self.sphere_radius

    def forward(
        self,
        sulc_subj: torch.Tensor,        # (B, 1, N_max) - sulc su mesh ico_out
        template_grid: torch.Tensor,    # (B, 1, H, W) o (1, H, W)
        sphere_init: torch.Tensor,      # (B, 3, N_max) - posizioni iniziali a ico_out
        return_intermediate: bool = False,
    ) -> torch.Tensor:
        """
        Returns
        -------
        r : (B, 3, N_max) posizioni 3D finali alla risoluzione ico_out.
        """

        # Iniziamo dalla risoluzione ico_in (prendiamo i primi N_in vertici)
        n_in = self.topologies[self.ico_in].n_vertices


        r = sphere_init[:, :, :n_in].clone()  # (B, 3, N_in)


        intermediate_stats = []  # Per salvare stats di dr se richiesto
 
        for order in self.active_orders:
            n_k = self.topologies[order].n_vertices
            unet = self.unets[str(order)]
            deform = self.deforms[str(order)]
  
        # Sulc subject alla scala corrente (primi n_k vertici)

            sulc_subj_k = sulc_subj[:, :, :n_k]  # (B, 1, n_k)

            for it in range(self.n_iter[order]):     # ← CAMBIATO: _ in it
            # Sample template grid alle posizioni correnti
                sulc_temp_at_r = sample_grid_at_vertices(template_grid, r)  # (B, 1, n_k)


            # Stack as 2 channels
                x = torch.cat([sulc_subj_k, sulc_temp_at_r], dim=1)  # (B, 2, n_k)

            # Predict deformation
                features = unet(x)            # (B, C, n_k)
                dr = deform(features)         # (B, 3, n_k)



            # Salva stats di dr se richiesto
                if return_intermediate:
                    intermediate_stats.append({
                        "order": order,
                        "iter": it,
                        "dr_mean": dr.abs().mean().item(),
                        "dr_max": dr.abs().max().item(),
                        "dr_std": dr.std().item(),
                        "n_verts": n_k,
                    })

            # Update positions and project to sphere

                r = self.project_to_sphere(r + self.step_size * dr)

        # Subdivide alla scala successiva (se non l'ultima)
            if order < self.ico_out:
                r = self.topologies[order].subdivide_vertices(r.mT).mT
                r = self.project_to_sphere(r)

        if return_intermediate:
            return r, intermediate_stats
        return r