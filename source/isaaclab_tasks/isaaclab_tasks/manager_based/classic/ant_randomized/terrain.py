"""Generated terrain with the original Ant grid and ground-adjusted spawn height."""

import math

import torch

from isaaclab.terrains import TerrainImporter
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh


class AntGridTerrainImporter(TerrainImporter):
    """Keep the stock grid rather than assigning several Ants to one tile center.

    Only enlarge the generated surface when the requested grid does not fit.
    Terrain shapes, tile size, sampling proportions and resolution are unchanged.
    """

    def __init__(self, cfg):
        if cfg.terrain_type == "generator":
            grid_rows = math.ceil(cfg.num_envs / int(math.sqrt(cfg.num_envs)))
            grid_cols = math.ceil(cfg.num_envs / grid_rows)
            generator = cfg.terrain_generator
            generator.num_rows = max(generator.num_rows, math.ceil(grid_rows * cfg.env_spacing / generator.size[0]))
            generator.num_cols = max(generator.num_cols, math.ceil(grid_cols * cfg.env_spacing / generator.size[1]))
        self._ground_mesh = None
        super().__init__(cfg)

    def import_mesh(self, name, mesh):
        super().import_mesh(name, mesh)
        self._ground_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
        self._ray_start_z = float(mesh.bounds[1, 2]) + 1.0

    def configure_env_origins(self, origins=None):
        # Use the exact grid formula used by the original plane terrain.
        super().configure_env_origins()
        if self._ground_mesh is not None:
            ground_z = self.ground_height(self.env_origins)
            if not torch.isfinite(ground_z).all():
                raise RuntimeError("Generated terrain does not cover every Ant grid origin.")
            self.env_origins[:, 2] = ground_z

    def ground_height(self, positions):
        """Query the static surface at XY; no robot sensor or policy input is added."""
        starts = positions.clone()
        starts[:, 2] = self._ray_start_z
        directions = torch.zeros_like(starts)
        directions[:, 2] = -1.0
        return raycast_mesh(starts, directions, self._ground_mesh)[0][:, 2]


def spawn_ant_above_terrain(prim_path, cfg, translation=None, orientation=None):
    """Avoid terrain penetration while physics initializes, before the first reset.

    Only the temporary spawned transform changes. Articulation default state still
    comes from Ant's original init_state, so every episode resets to ground + 0.5 m.
    """
    from pxr import Usd, UsdGeom

    import isaaclab.sim as sim_utils

    ground = sim_utils.get_current_stage().GetPrimAtPath("/World/ground")
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(ground)
    top = float(bounds.ComputeAlignedRange().GetMax()[2])
    x, y, z = translation
    return sim_utils.spawn_from_usd(
        prim_path, cfg, translation=(x, y, z + max(0.0, top)), orientation=orientation
    )
