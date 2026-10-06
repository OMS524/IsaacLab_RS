"""Keep baseline PPO hyperparameters and separate experiment directories."""

from isaaclab.utils import configclass

from ..ant.agents.rsl_rl_ppo_cfg import AntPPORunnerCfg


@configclass
class AntRandomizedPPORunnerCfg(AntPPORunnerCfg):
    experiment_name = "ant_dr"


@configclass
class AntRandomizedPushPPORunnerCfg(AntPPORunnerCfg):
    experiment_name = "ant_dr_push"


@configclass
class AntRandomizedStablePPORunnerCfg(AntPPORunnerCfg):
    experiment_name = "ant_dr_stable"


@configclass
class AntRandomizedForwardPPORunnerCfg(AntPPORunnerCfg):
    experiment_name = "ant_dr_forward"


@configclass
class AntRandomizedForwardLightPPORunnerCfg(AntRandomizedForwardPPORunnerCfg):
    experiment_name = "ant_dr_forward_light"


@configclass
class AntFinetunePPORunnerCfg(AntPPORunnerCfg):
    # All three original-reward stages share a root for train.py --load_run.
    experiment_name = "ant_ft_original"


@configclass
class AntFinetuneLightPPORunnerCfg(AntPPORunnerCfg):
    experiment_name = "ant_ft_forward_light"
