"""Ant domain-randomization experiments; the original Isaac-Ant-v0 is untouched."""

import gymnasium as gym

for task_id, env_class, runner_class in (
    ("Isaac-Ant-DR-v0", "AntRandomizedEnvCfg", "AntRandomizedPPORunnerCfg"),
    ("Isaac-Ant-DR-Stable-v0", "AntRandomizedStableEnvCfg", "AntRandomizedStablePPORunnerCfg"),
    ("Isaac-Ant-DR-Forward-v0", "AntRandomizedForwardEnvCfg", "AntRandomizedForwardPPORunnerCfg"),
    ("Isaac-Ant-DR-Forward-Light-v0", "AntRandomizedForwardLightEnvCfg", "AntRandomizedForwardLightPPORunnerCfg"),
    ("Isaac-Ant-DR-Push-v0", "AntRandomizedPushEnvCfg", "AntRandomizedPushPPORunnerCfg"),
    ("Isaac-Ant-DR-Eval-v0", "AntRandomizedEvalEnvCfg", "AntRandomizedPPORunnerCfg"),
    ("Isaac-Ant-DR-Eval-Push-v0", "AntRandomizedEvalPushEnvCfg", "AntRandomizedPushPPORunnerCfg"),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_env_cfg:{env_class}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents_cfg:{runner_class}",
        },
    )


# Shared log roots allow the stock train.py resume path to cross terrain stages.
# The final stage reuses the existing full-DR classes without changing their settings.
for task_id, env_module, env_class, runner_class in (
    ("Isaac-Ant-FT-Flat-v0", "finetune_env_cfg", "AntFinetuneFlatEnvCfg", "AntFinetunePPORunnerCfg"),
    ("Isaac-Ant-FT-Mild-v0", "finetune_env_cfg", "AntFinetuneMildEnvCfg", "AntFinetunePPORunnerCfg"),
    ("Isaac-Ant-FT-DR-v0", "ant_env_cfg", "AntRandomizedEnvCfg", "AntFinetunePPORunnerCfg"),
    ("Isaac-Ant-FT-Light-Flat-v0", "finetune_env_cfg", "AntFinetuneLightFlatEnvCfg", "AntFinetuneLightPPORunnerCfg"),
    ("Isaac-Ant-FT-Light-Mild-v0", "finetune_env_cfg", "AntFinetuneLightMildEnvCfg", "AntFinetuneLightPPORunnerCfg"),
    ("Isaac-Ant-FT-Light-DR-v0", "ant_env_cfg", "AntRandomizedForwardLightEnvCfg", "AntFinetuneLightPPORunnerCfg"),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{env_module}:{env_class}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents_cfg:{runner_class}",
        },
    )


# Use the baseline log root "ant" so stock train.py can resume the exact old run.
# New run_name values keep the two fine-tuned outputs in distinct directories.
for task_id, env_class in (
    ("Isaac-Ant-Baseline-FT-v0", "AntBaselineFinetuneEnvCfg"),
    ("Isaac-Ant-Baseline-FT-Light-v0", "AntBaselineFinetuneLightEnvCfg"),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.baseline_finetune_env_cfg:{env_class}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents_cfg:AntPPORunnerCfg",
        },
    )


# Every policy is evaluated with the original seven rewards on fresh block layouts.
gym.register(
    id="Isaac-Ant-DR-Eval-Blocks-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.block_eval_env_cfg:AntBlockEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents_cfg:AntPPORunnerCfg",
    },
)
