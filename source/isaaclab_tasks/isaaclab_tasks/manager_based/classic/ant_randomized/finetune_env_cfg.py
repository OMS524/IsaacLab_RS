"""Three-stage Ant training, paired by original and Forward-Light rewards.

Keep rewards, observations, actions and PPO fixed within each branch. Only
terrain and contact friction change: flat -> mild DR -> existing full DR.
"""

from isaaclab.utils import configclass

from isaaclab.terrains import TerrainImporter
from isaaclab.sim import spawn_from_usd

from ..ant.ant_env_cfg import TerminationsCfg
from .ant_env_cfg import AntRandomizedEnvCfg, AntRandomizedForwardLightEnvCfg


def _configure_flat_stage(cfg):
    """Original flat-ground placement and termination, with fixed contact friction."""
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.class_type = TerrainImporter
    cfg.scene.robot.spawn.func = spawn_from_usd
    cfg.terminations = TerminationsCfg()
    cfg.events.contact_friction.params.update(
        static_friction_range=(1.0, 1.0),
        dynamic_friction_range=(1.0, 1.0),
        num_buckets=1,
    )


def _configure_mild_stage(cfg):
    """Retain flat ground while narrowing terrain height/slope and friction ranges."""
    terrains = cfg.scene.terrain.terrain_generator.sub_terrains
    terrains["flat"].proportion = 0.5
    for name in ("slope", "inverted_slope"):
        terrains[name].proportion = 0.1
        terrains[name].slope_range = (0.01, 0.04)
    terrains["rough"].proportion = 0.3
    terrains["rough"].noise_range = (-0.01, 0.01)
    cfg.events.contact_friction.params.update(
        static_friction_range=(0.8, 1.1),
        dynamic_friction_range=(0.7, 1.0),
    )


@configclass
class AntFinetuneFlatEnvCfg(AntRandomizedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_flat_stage(self)


@configclass
class AntFinetuneMildEnvCfg(AntRandomizedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_mild_stage(self)


@configclass
class AntFinetuneLightFlatEnvCfg(AntRandomizedForwardLightEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_flat_stage(self)


@configclass
class AntFinetuneLightMildEnvCfg(AntRandomizedForwardLightEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_mild_stage(self)
