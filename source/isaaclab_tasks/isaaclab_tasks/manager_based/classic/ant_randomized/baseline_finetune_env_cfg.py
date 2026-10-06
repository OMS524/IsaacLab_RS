"""One-stage fine-tuning from a shared flat Ant checkpoint on mixed terrain."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.height_field.hf_terrains import discrete_obstacles_terrain
from isaaclab.utils import configclass

from .ant_env_cfg import AntRandomizedEnvCfg, AntRandomizedForwardLightEnvCfg, training_terrain


# Retained for archived environment configurations that reference this function.
def rectangular_blocks_terrain(difficulty, cfg):
    """Keep block edges vertical without changing the rough/slope mesh settings.

    TerrainGenerator overwrites per-terrain slope thresholds, so apply the block
    threshold here on a copy when calling the built-in height-field generator.
    """
    block_cfg = cfg.copy()
    block_cfg.slope_threshold = 0.0
    return discrete_obstacles_terrain(difficulty, block_cfg)


def baseline_finetune_terrain() -> terrain_gen.TerrainGeneratorCfg:
    """10% flat, 30% random roughness, 40% random-grid cells, 20% shallow slopes."""
    cfg = training_terrain()
    cfg.sub_terrains["flat"].proportion = 0.1
    cfg.sub_terrains["rough"].proportion = 0.3
    cfg.sub_terrains["slope"].proportion = 0.1
    cfg.sub_terrains["inverted_slope"].proportion = 0.1
    cfg.sub_terrains["blocks"] = terrain_gen.MeshRandomGridTerrainCfg(
        proportion=0.4,
        # Match the existing Blocks evaluation terrain; retain a positive tile border.
        grid_width=0.6,
        grid_height_range=(0.06, 0.06),
        platform_width=2.0,
        holes=False,
    )
    return cfg


@configclass
class AntBaselineFinetuneEnvCfg(AntRandomizedEnvCfg):
    """Original seven rewards, flat/roughness/grid/slopes, no pushes."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = baseline_finetune_terrain()


@configclass
class AntBaselineFinetuneLightEnvCfg(AntRandomizedForwardLightEnvCfg):
    """Identical terrain and dynamics with relaxed vertical/heading penalties."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = baseline_finetune_terrain()
        self.rewards.vertical_velocity.weight = -0.05
        self.rewards.heading_error.weight = -0.1
