"""Paired Ant experiments: identical terrain/friction, with or without pushes."""

import isaaclab.terrains as terrain_gen
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..ant.ant_env_cfg import AntEnvCfg, EventCfg, RewardsCfg, TerminationsCfg
from . import mdp
from .terrain import AntGridTerrainImporter, spawn_ant_above_terrain


def training_terrain() -> terrain_gen.TerrainGeneratorCfg:
    """A reproducible bank of flat, shallow sloped, and mildly uneven tiles."""
    return terrain_gen.TerrainGeneratorCfg(
        seed=1701,
        curriculum=False,
        size=(20.0, 20.0),
        num_rows=8,
        num_cols=8,
        border_width=0.0,
        horizontal_scale=0.2,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
            "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.15, slope_range=(0.02, 0.10), platform_width=2.0, border_width=0.4
            ),
            "inverted_slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.15, slope_range=(0.02, 0.10), platform_width=2.0, border_width=0.4
            ),
            "rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.4,
                noise_range=(-0.025, 0.025),
                noise_step=0.005,
                downsampled_scale=0.4,
                border_width=0.4,
            ),
        },
    )


def evaluation_terrain() -> terrain_gen.TerrainGeneratorCfg:
    """Held-out wave geometry, absent from the training bank; not a harder training seed."""
    cfg = training_terrain()
    cfg.seed = 2401
    cfg.sub_terrains = {
        "waves_long": terrain_gen.HfWaveTerrainCfg(
            proportion=0.5, amplitude_range=(0.03, 0.06), num_waves=4, border_width=0.4
        ),
        "waves_short": terrain_gen.HfWaveTerrainCfg(
            proportion=0.5, amplitude_range=(0.03, 0.06), num_waves=8, border_width=0.4
        ),
    }
    return cfg

@configclass
class AntRandomizedEventsCfg(EventCfg):
    contact_friction = EventTerm(
        func=mdp.randomize_contact_friction,
        mode="startup",
        params={
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.4, 1.0),
            "num_buckets": 64,
        },
    )
    push_robot: EventTerm | None = None

@configclass
class AntRandomizedTerminationsCfg(TerminationsCfg):
    torso_height = DoneTerm(
        func=mdp.root_height_below_terrain,
        params={"minimum_height": 0.31},
    )


def push_event() -> EventTerm:
    """A brief horizontal velocity increment, not a sustained force in Newtons."""
    return EventTerm(
        func=base_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        is_global_time=False,
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )

@configclass
class AntRandomizedEnvCfg(AntEnvCfg):
    events: AntRandomizedEventsCfg = AntRandomizedEventsCfg()
    terminations: AntRandomizedTerminationsCfg = AntRandomizedTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = training_terrain()
        # Generated terrain normally samples shared tile centers. Preserve Ant's
        # original grid instead, correcting only its height for the local ground.
        self.scene.terrain.class_type = AntGridTerrainImporter
        # Physics initializes before episode reset; keep the temporary spawn above
        # the mesh while preserving the original robot init_state and reset event.
        self.scene.robot.spawn.func = spawn_ant_above_terrain


@configclass
class AntStableRewardsCfg(RewardsCfg):
    """Add two training penalties to the original seven Ant reward terms.

    RewardManager applies weight * dt; these functions return unweighted squares.
    Evaluation tasks continue to inherit the original RewardsCfg.
    """

    vertical_velocity = RewTerm(func=mdp.world_lin_vel_z_l2, weight=-0.5)
    roll_pitch_velocity = RewTerm(func=base_mdp.ang_vel_xy_l2, weight=-0.05)

@configclass
class AntRandomizedStableEnvCfg(AntRandomizedEnvCfg):
    """No-push DR training with vertical and body roll/pitch velocity penalties."""

    rewards: AntStableRewardsCfg = AntStableRewardsCfg()

@configclass
class AntForwardRewardsCfg(AntStableRewardsCfg):
    """Keep the original and stability terms, adding sideways/heading penalties."""

    lateral_velocity = RewTerm(func=mdp.body_lin_vel_y_l2, weight=-0.25)
    heading_error = RewTerm(
        func=mdp.heading_error_cost, weight=-1.0, params={"target_pos": (1000.0, 0.0, 0.0)}
    )

@configclass
class AntRandomizedForwardEnvCfg(AntRandomizedStableEnvCfg):
    """Stable DR training with additional body-forward motion shaping."""

    rewards: AntForwardRewardsCfg = AntForwardRewardsCfg()

@configclass
class AntRandomizedForwardLightEnvCfg(AntRandomizedForwardEnvCfg):
    """Change only the body-lateral velocity weight, from -0.25 to -0.05."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.lateral_velocity.weight = -0.05

@configclass
class AntRandomizedPushEnvCfg(AntRandomizedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = push_event()

@configclass
class AntRandomizedEvalEnvCfg(AntRandomizedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 100
        self.scene.terrain.terrain_generator = evaluation_terrain()

@configclass
class AntRandomizedEvalPushEnvCfg(AntRandomizedEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = push_event()
