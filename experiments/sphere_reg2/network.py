"""
Rete di registrazione sferica semplice.

Architettura:
  - Una graph UNet (DeepSurferTopology, ico6 = 245762 vertici)
  - Input: 2 canali (sulc_subj, sulc_temp_resampled)
  - Output: campo di deformazione 3D
  - Iterata N_ITER volte alla stessa risoluzione

Forward:
  r = sphere_subj_init.clone()
  for _ in range(N_ITER):
      sulc_temp_at_r = sample_grid(template_grid, r)
      x = stack(sulc_subj, sulc_temp_at_r)        # (B, 2, N)
      dr = unet(x)                                  # (B, 3, N)
      r = project_to_sphere(r + step * dr)
  return r
"""

import torch
import torch.nn.functional as F

from brainnet.mesh.topology import DeepSurferTopology
from brainnet.modules.graph.layers import EdgeConvolutionBlock, EdgeConvolution
from brainnet.modules.graph.unet import UNet

from experiments.sphere_reg2.dataset import sample_grid_at_vertices


class SphereRegNet(torch.nn.Module):
    """Graph UNet che predice deformazioni su una sfera ico6 (245762 vertici).

    Parameters
    ----------
    n_iter : int
        Numero di iterazioni della rete alla stessa risoluzione.
    ico_order : int
        Ordine icosaedrale del DeepSurferTopology (default 6 = 245762 vertici).
    in_channels : int
        Canali di input (2 = sulc_subj + sulc_temp_resampled).
    base_channels : int
        Larghezza dell'encoder/decoder UNet.
    sphere_radius : float
        Raggio della sfera.
    step_size : float
        Scala del campo di deformazione applicato ad ogni iterazione.
    """

    def __init__(
        self,
        n_iter: int = 1,
        ico_order: int = 6,
        in_channels: int = 2,
        base_channels: int = 64,
        sphere_radius: float = 100.0,
        step_size: float = 1.0,
    ):
        super().__init__()
        self.n_iter = n_iter
        self.sphere_radius = sphere_radius
        self.step_size = step_size

        self.topologies = DeepSurferTopology.recursive_subdivision(ico_order)

        self.unet = UNet(
            in_channels=in_channels,
            topologies=self.topologies,
            conv_module=EdgeConvolutionBlock,
            reduce="amax",
            channels=base_channels,
            max_depth=4,
            n_conv=1,
        )

        # Final convolution: features -> 3D deformation
        last_topo = self.topologies[-1]
        self.deform = EdgeConvolution(
            in_channels=self.unet.out_ch,
            out_channels=3,
            reduce_index=last_topo.conv_index_reduce,
            gather_index=last_topo.conv_index_gather,
            bias=False,
            init_values=0.0,
        )

    #def project_to_sphere(self, r: torch.Tensor) -> torch.Tensor:
        #"""Project (B, 3, N) onto sphere of radius `sphere_radius`."""
        #return F.normalize(r, dim=1) * self.sphere_radius

    def project_to_sphere(self, r: torch.Tensor) -> torch.Tensor:
        """Project (B, 3, N) onto sphere, robusto a vettori vicini a zero."""
        norm = r.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return (r / norm) * self.sphere_radius
        
    # After the deformation update, we project the vertices back onto the sphere 
    # to ensure they remain on the surface using this project_to_sphere funtion.



    def forward(
        self,
        sulc_subj: torch.Tensor,        # (B, 1, N)
        template_grid: torch.Tensor,    # (B, 1, H, W) or (1, H, W)
        sphere_init: torch.Tensor,      # (B, 3, N) posizioni iniziali
    ) -> torch.Tensor:
        """
        Returns
        -------
        r : (B, 3, N) posizioni 3D finali dei vertici sulla sfera.
        """
        r = sphere_init.clone()

        for _ in range(self.n_iter):
            # Sample template grid at current vertex positions
            sulc_temp_at_r = sample_grid_at_vertices(template_grid, r)  # (B, 1, N)

            # Stack as 2 channels
            x = torch.cat([sulc_subj, sulc_temp_at_r], dim=1)  # (B, 2, N)

            # Predict deformation
            features = self.unet(x)            # (B, C, N)
            dr = self.deform(features)         # (B, 3, N)
            dr = dr.clamp(-10.0, 10.0)

            # Update positions and project to sphere
            r = self.project_to_sphere(r + self.step_size * dr)

        return r