"""Random-grid evaluation using the original seven Ant rewards.

Replace sparse rectangular obstacles with dense square cells of random height.
Keep the existing Blocks task ID so evaluation commands remain compatible.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from .ant_env_cfg import AntRandomizedEvalEnvCfg, training_terrain


def block_evaluation_terrain():
    """Use 0.6 m square grid cells with uniformly sampled heights within +/-6 cm."""
    cfg = training_terrain()
    cfg.seed = 2402
    cfg.sub_terrains = {
        "blocks": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0,
            # Leave a positive border within each 20 m tile, as the generator requires.
            grid_width=0.6,
            grid_height_range=(0.06, 0.06),
            platform_width=2.0,
            holes=False,
        ),
    }
    return cfg


@configclass
class AntBlockEvalEnvCfg(AntRandomizedEvalEnvCfg):
    """Original rewards, 100 environments, unchanged friction, and no pushes."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = block_evaluation_terrain()
