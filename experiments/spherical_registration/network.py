"""
Network with rotations of dr vector, to keep points on the sphere without needing a separate projection step.
"""

import torch
import torch.nn.functional as F

from brainnet.mesh.topology import DeepSurferTopology
from brainnet.modules.graph.layers import EdgeConvolutionBlock, EdgeConvolution
from brainnet.modules.graph.unet import UNet
from brainnet.mesh.surface import rotate

from experiments.spherical_registration.dataset import sample_grid_at_vertices
 

def solve_ode_euler_rotate(h, v, dv):
    """Rotate v about the (normalized) vector dv by an angle of h * |dv|.

    v, dv : (B, 3, N)
    h     : float (step size)

    Returns : (B, 3, N) rotated on the sphere (radius preserved by construction).
    """
    v = v.mT       # (B, N, 3)
    dv = dv.mT     # (B, N, 3)
    theta = torch.linalg.vector_norm(dv, dim=2, keepdim=True, dtype=dv.dtype)
    k = dv / (theta + 1e-8)
    return rotate(v, k, h * theta, normalize=False).mT  # (B, 3, N)


class SphereRegNet(torch.nn.Module):
    """Coarse-to-fine spherical registration."""

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

        self.topologies = DeepSurferTopology.recursive_subdivision(ico_out)
        self.active_orders = list(range(ico_in, ico_out + 1))

        if isinstance(n_iter, int):
            n_iter = [n_iter] * len(self.active_orders)
        assert len(n_iter) == len(self.active_orders), \
            f"n_iter must have length {len(self.active_orders)}"
        self.n_iter = dict(zip(self.active_orders, n_iter))

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
                init_values=0.01, 

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
        sulc_subj: torch.Tensor,
        template_grid: torch.Tensor,
        sphere_init: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor:
        """
        Returns
        -------
        r : (B, 3, N_max) 3D final positions at resolution ico_out.
        """
        n_in = self.topologies[self.ico_in].n_vertices
        r = sphere_init[:, :, :n_in].clone()

        intermediate_stats = []

        for order in self.active_orders:
            n_k = self.topologies[order].n_vertices
            unet = self.unets[str(order)]
            deform = self.deforms[str(order)]

            sulc_subj_k = sulc_subj[:, :, :n_k]

            for it in range(self.n_iter[order]):
                sulc_temp_at_r = sample_grid_at_vertices(template_grid, r)
                x = torch.cat([sulc_subj_k, sulc_temp_at_r], dim=1)

                features = unet(x)
                dr = deform(features)
                dr = dr.clamp(-10, 10)  



                if return_intermediate:
                    intermediate_stats.append({
                        "order": order,
                        "iter": it,
                        "dr_mean": dr.abs().mean().item(),
                        "dr_max": dr.abs().max().item(),
                        "dr_std": dr.std().item(),
                        "n_verts": n_k,
                    })

                r = solve_ode_euler_rotate(self.step_size, r, dr)
                r = F.normalize(r, dim=1) * self.sphere_radius

            # Subdivision for next level, if any
            if order < self.ico_out:
                r = self.topologies[order].subdivide_vertices(r.mT).mT
                r = self.project_to_sphere(r)

        if return_intermediate:
            return r, intermediate_stats
        return r