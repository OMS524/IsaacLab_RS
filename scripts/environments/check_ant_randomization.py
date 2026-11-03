"""GPU smoke checks for the paired Ant DR tasks, without training or changing the baseline.

Run from IsaacLab_RS:
    ./isaaclab.sh -p scripts/environments/check_ant_randomization.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Ant-DR-Push-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch
from types import SimpleNamespace

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import mdp as base_mdp
from isaaclab.utils import math as math_utils
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
from isaaclab_tasks.manager_based.classic.ant_randomized import mdp
from isaaclab_tasks.manager_based.classic.ant_randomized.ant_env_cfg import (
    AntRandomizedEnvCfg,
    AntRandomizedForwardEnvCfg,
    AntRandomizedForwardLightEnvCfg,
    AntRandomizedEvalEnvCfg,
    AntRandomizedEvalPushEnvCfg,
    AntRandomizedPushEnvCfg,
    AntRandomizedStableEnvCfg,
)
from isaaclab_tasks.manager_based.classic.ant_randomized.agents_cfg import (
    AntRandomizedPPORunnerCfg,
    AntRandomizedForwardPPORunnerCfg,
    AntRandomizedForwardLightPPORunnerCfg,
    AntRandomizedPushPPORunnerCfg,
    AntRandomizedStablePPORunnerCfg,
)
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def check_robot_config(cfg, baseline):
    from isaaclab_tasks.manager_based.classic.ant_randomized.terrain import spawn_ant_above_terrain

    robot = cfg.scene.robot.to_dict()
    original = baseline.scene.robot.to_dict()
    if cfg.scene.terrain.terrain_type == "generator":
        assert cfg.scene.robot.spawn.func is spawn_ant_above_terrain
        robot["spawn"]["func"] = original["spawn"]["func"]
    assert robot == original, "Only the temporary pre-reset spawn transform may differ"


def check_configs():
    baseline = AntEnvCfg()
    classes = (AntRandomizedEnvCfg, AntRandomizedPushEnvCfg, AntRandomizedEvalEnvCfg, AntRandomizedEvalPushEnvCfg)
    configs = [cls() for cls in classes]
    for cfg in configs:
        cfg.validate()
        for key in ("rewards", "observations", "actions"):
            assert getattr(cfg, key).to_dict() == getattr(baseline, key).to_dict(), key
        assert cfg.episode_length_s == baseline.episode_length_s == 16.0
        check_robot_config(cfg, baseline)
        assert cfg.viewer.to_dict() == baseline.viewer.to_dict()
        assert cfg.scene.clone_in_fabric == baseline.scene.clone_in_fabric
        assert cfg.scene.env_spacing == baseline.scene.env_spacing
        assert cfg.scene.replicate_physics == baseline.scene.replicate_physics
        assert cfg.scene.filter_collisions == baseline.scene.filter_collisions
        assert cfg.scene.terrain.physics_material.to_dict() == baseline.scene.terrain.physics_material.to_dict()
        assert cfg.events.reset_base.to_dict() == baseline.events.reset_base.to_dict()
    for i, j in ((0, 1), (2, 3)):
        plain, pushed = configs[i].to_dict(), configs[j].to_dict()
        pushed["events"]["push_robot"] = None
        assert plain == pushed, "Paired tasks must differ only in the push event"
    a, b = AntRandomizedPPORunnerCfg().to_dict(), AntRandomizedPushPPORunnerCfg().to_dict()
    b["experiment_name"] = a["experiment_name"]
    assert a == b, "Paired tasks must use identical PPO settings"
    stable = AntRandomizedStableEnvCfg()
    stable.validate()
    stable_dict = stable.to_dict()
    stability_terms = {"vertical_velocity", "roll_pitch_velocity"}
    assert set(stable_dict["rewards"]) == set(baseline.rewards.to_dict()) | stability_terms
    assert stable.rewards.vertical_velocity.weight == -0.5
    assert stable.rewards.roll_pitch_velocity.weight == -0.05
    for name in stability_terms:
        del stable_dict["rewards"][name]
    assert stable_dict == configs[0].to_dict(), "Stable task must differ from DR only by its two penalties"
    stable_agent = AntRandomizedStablePPORunnerCfg().to_dict()
    assert stable_agent["experiment_name"] == "ant_dr_stable"
    stable_agent["experiment_name"] = a["experiment_name"]
    assert stable_agent == a, "Reward experiments must use identical PPO settings"
    forward = AntRandomizedForwardEnvCfg()
    forward.validate()
    forward_dict = forward.to_dict()
    forward_terms = {"lateral_velocity", "heading_error"}
    assert set(forward_dict["rewards"]) == set(stable.rewards.to_dict()) | forward_terms
    assert forward.rewards.lateral_velocity.weight == -0.25
    assert forward.rewards.heading_error.weight == -1.0
    assert forward.rewards.heading_error.params["target_pos"] == baseline.rewards.progress.params["target_pos"]
    assert forward.rewards.heading_error.params["target_pos"] == baseline.rewards.move_to_target.params["target_pos"]
    for name in forward_terms:
        del forward_dict["rewards"][name]
    assert forward_dict == stable.to_dict(), "Forward task must add only two penalties to Stable"
    forward_agent = AntRandomizedForwardPPORunnerCfg().to_dict()
    assert forward_agent["experiment_name"] == "ant_dr_forward"
    forward_agent["experiment_name"] = a["experiment_name"]
    assert forward_agent == a, "Forward must preserve PPO settings"
    light = AntRandomizedForwardLightEnvCfg()
    light.validate()
    assert light.rewards.lateral_velocity.weight == -0.05
    light_dict = light.to_dict()
    light_dict["rewards"]["lateral_velocity"]["weight"] = -0.25
    assert light_dict == forward.to_dict(), "Light must change only the lateral weight"
    assert AntRandomizedForwardEnvCfg().rewards.lateral_velocity.weight == -0.25
    light_agent = AntRandomizedForwardLightPPORunnerCfg().to_dict()
    assert light_agent["experiment_name"] == "ant_dr_forward_light"
    light_agent["experiment_name"] = a["experiment_name"]
    assert light_agent == a, "Light must preserve PPO settings"
    forward.rewards.vertical_velocity.weight = -99.0
    assert stable.rewards.vertical_velocity.weight == -0.5, "Reward configurations must be isolated"
    stable.rewards.progress.weight = 99.0
    assert configs[0].rewards.progress.weight == configs[2].rewards.progress.weight == baseline.rewards.progress.weight
    # World-horizontal motion must have no vertical penalty, even when body z differs.
    # Positive/negative vertical speeds have equal cost. Pure yaw has no roll/pitch cost.
    velocities = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(
        root_lin_vel_w=torch.tensor([[3., 4., 0.], [0., 0., 2.], [0., 0., -2.]]),
        root_lin_vel_b=torch.tensor([[0., 0., 5.], [2., 0., 0.], [-2., 0., 0.]]),
        root_ang_vel_b=torch.tensor([[0., 0., 6.], [1., 2., 8.], [-1., -2., 0.]]),
    ))})
    assert mdp.world_lin_vel_z_l2(velocities).tolist() == [0., 4., 4.]
    assert base_mdp.ang_vel_xy_l2(velocities).tolist() == [0., 5., 5.]
    # Forward motion after a 90-degree yaw must not count as body-lateral motion.
    yaw = torch.tensor([0., torch.pi / 2, torch.pi / 2, -torch.pi / 2])
    quat = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    world_velocity = torch.tensor([[5., 0., 0.], [0., 5., 0.], [3., 0., 0.], [3., 0., 0.]])
    lateral = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(
        root_lin_vel_w=world_velocity,
        root_lin_vel_b=math_utils.quat_apply_inverse(quat, world_velocity),
    ))})
    assert torch.allclose(mdp.body_lin_vel_y_l2(lateral), torch.tensor([0., 0., 9., 9.]), atol=1e-5)
    # Aligned, perpendicular, reversed, angle-wrap, off-axis target, and exactly-at-goal cases.
    heading = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.tensor([
            [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
            [1001., 0., 0.], [1001., 0., 0.], [1000., -5., 8.], [1000., 0., 8.],
        ]),
        heading_w=torch.tensor([
            0., torch.pi / 2, torch.pi, -torch.pi / 2,
            torch.pi - 0.1, -torch.pi + 0.1, torch.pi / 2, torch.pi,
        ]),
    ))})
    heading_cost = mdp.heading_error_cost(heading, (1000., 0., 0.))
    assert torch.allclose(heading_cost, torch.tensor([0., 1., 2., 1., .004995835, .004995835, 0., 0.]), atol=1e-6)
    assert torch.isfinite(heading_cost).all() and ((heading_cost >= 0) & (heading_cost <= 2)).all()
    train_types = {type(c) for c in configs[0].scene.terrain.terrain_generator.sub_terrains.values()}
    eval_types = {type(c) for c in configs[2].scene.terrain.terrain_generator.sub_terrains.values()}
    assert train_types.isdisjoint(eval_types), "Evaluation geometry must be held out"
    configs[0].scene.terrain.terrain_generator.sub_terrains.clear()
    assert configs[1].scene.terrain.terrain_generator.sub_terrains, "Configs must not share mutable state"
    assert baseline.scene.terrain.terrain_type == "plane"
    for task in (
        "Isaac-Ant-DR-v0",
        "Isaac-Ant-DR-Push-v0",
        "Isaac-Ant-DR-Eval-v0",
        "Isaac-Ant-DR-Eval-Push-v0",
        "Isaac-Ant-DR-Stable-v0",
        "Isaac-Ant-DR-Forward-v0",
        "Isaac-Ant-DR-Forward-Light-v0",
    ):
        assert gym.spec(task)
    # A robot over a depression is healthy; a robot high in world coordinates can still have fallen.
    roots = torch.tensor([[0., 0., -0.5], [0., 0., 2.2], [0., 0., 1.], [0., 0., 0.2]])
    hits = torch.tensor([[[0., 0., -1.]], [[0., 0., 2.]], [[0., 0., float("inf")]], [[0., 0., float("inf")]]])
    class FakeScene(dict):
        terrain = SimpleNamespace(ground_height=lambda positions: hits[:, 0, 2])
    fake = SimpleNamespace(scene=FakeScene(robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=roots))))
    assert mdp.root_height_below_terrain(fake, 0.31).tolist() == [False, True, False, True]
    print(
        "[CHECK] Config isolation, original evaluation rewards, reward frames/signs, heading wrap/goal cases: PASS",
        flush=True,
    )


def check_finetune_configs():
    baseline = AntEnvCfg()
    full = AntRandomizedEnvCfg()
    light = AntRandomizedForwardLightEnvCfg()
    configs = {}
    for suffix, expected_rewards, experiment in (
        ("", baseline.rewards, "ant_ft_original"),
        ("Light-", light.rewards, "ant_ft_forward_light"),
    ):
        for stage in ("Flat", "Mild", "DR"):
            task = f"Isaac-Ant-FT-{suffix}{stage}-v0"
            cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
            cfg.validate()
            assert cfg.rewards.to_dict() == expected_rewards.to_dict(), task
            for key in ("observations", "actions"):
                assert getattr(cfg, key).to_dict() == getattr(baseline, key).to_dict(), task
            check_robot_config(cfg, baseline)
            assert cfg.events.push_robot is None
            assert cfg.episode_length_s == baseline.episode_length_s
            agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point").to_dict()
            assert agent["experiment_name"] == experiment
            agent["experiment_name"] = "ant_dr"
            assert agent == AntRandomizedPPORunnerCfg().to_dict(), task
            configs[suffix, stage] = cfg
        flat = configs[suffix, "Flat"]
        assert flat.scene.terrain.terrain_type == "plane"
        assert flat.scene.terrain.terrain_generator is None
        assert flat.terminations.to_dict() == baseline.terminations.to_dict()
        assert flat.events.reset_base.to_dict() == baseline.events.reset_base.to_dict()
        assert flat.events.contact_friction.params["static_friction_range"] == (1., 1.)
        assert flat.events.contact_friction.params["dynamic_friction_range"] == (1., 1.)
        mild = configs[suffix, "Mild"]
        tiles = mild.scene.terrain.terrain_generator.sub_terrains
        assert tiles["rough"].noise_range == (-.01, .01)
        assert tiles["rough"].proportion == .3
        assert tiles["flat"].proportion == .5
        assert tiles["slope"].slope_range == tiles["inverted_slope"].slope_range == (.01, .04)
        assert mild.events.contact_friction.params["static_friction_range"] == (.8, 1.1)
        assert mild.events.contact_friction.params["dynamic_friction_range"] == (.7, 1.)
    for stage in ("Flat", "Mild", "DR"):
        a, b = (configs[suffix, stage].to_dict() for suffix in ("", "Light-"))
        b["rewards"] = a["rewards"]
        assert a == b, f"Paired {stage} tasks must differ only in rewards"
    assert configs["", "DR"].to_dict() == full.to_dict()
    assert configs["Light-", "DR"].to_dict() == light.to_dict()
    configs["", "Mild"].scene.terrain.terrain_generator.sub_terrains["rough"].noise_range = (-99., 99.)
    assert configs["Light-", "Mild"].scene.terrain.terrain_generator.sub_terrains["rough"].noise_range == (-.01, .01)
    assert full.scene.terrain.terrain_generator.sub_terrains["rough"].noise_range == (-.025, .025)
    assert AntRandomizedEvalEnvCfg().rewards.to_dict() == baseline.rewards.to_dict()
    print("[CHECK] Six fine-tuning tasks: paired rewards, fixed PPO, isolated stages, unchanged full DR/Eval: PASS", flush=True)


def check_baseline_finetune_configs():
    import numpy as np
    from isaaclab_tasks.manager_based.classic.ant_randomized.baseline_finetune_env_cfg import (
        AntBaselineFinetuneEnvCfg, AntBaselineFinetuneLightEnvCfg,
    )
    original, light = AntBaselineFinetuneEnvCfg(), AntBaselineFinetuneLightEnvCfg()
    original.validate()
    light.validate()
    assert original.rewards.to_dict() == AntEnvCfg().rewards.to_dict()
    assert light.rewards.to_dict() == AntRandomizedForwardLightEnvCfg().rewards.to_dict()
    a, b = original.to_dict(), light.to_dict()
    b["rewards"] = a["rewards"]
    assert a == b, "Baseline fine-tunes must differ only in rewards"
    a["scene"]["terrain"]["terrain_generator"] = AntRandomizedEnvCfg().scene.terrain.terrain_generator.to_dict()
    assert a == AntRandomizedEnvCfg().to_dict(), "Only terrain changes from existing DR"
    generator = original.scene.terrain.terrain_generator
    terrains = generator.sub_terrains
    assert set(terrains) == {"rough", "blocks", "slope", "inverted_slope"}
    assert [terrains[k].proportion for k in ("rough", "blocks", "slope", "inverted_slope")] == [.4, .3, .15, .15]
    assert original.events.push_robot is None
    assert generator.curriculum is False
    for task in ("Isaac-Ant-Baseline-FT-v0", "Isaac-Ant-Baseline-FT-Light-v0"):
        agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point").to_dict()
        assert agent["experiment_name"] == "ant", "The old checkpoint must be discoverable by stock train.py"
        agent["experiment_name"] = "ant_dr"
        assert agent == AntRandomizedPPORunnerCfg().to_dict()
    # Validate the actual block mesh, including height and abrupt vertical edges.
    blocks = terrains["blocks"].copy()
    for key in ("size", "horizontal_scale", "vertical_scale", "slope_threshold"):
        setattr(blocks, key, getattr(generator, key))
    np_state = np.random.get_state()
    try:
        np.random.seed(24)
        meshes, origin = blocks.function(.5, blocks)
    finally:
        np.random.set_state(np_state)
    mesh = meshes[0]
    assert np.isfinite(mesh.vertices).all() and np.isfinite(origin).all()
    assert mesh.vertices[:, 2].min() >= -.020001 and mesh.vertices[:, 2].max() <= .020001
    assert mesh.vertices[:, 2].min() < 0 < mesh.vertices[:, 2].max()
    assert ((np.abs(mesh.face_normals[:, 2]) < 1e-6) & (np.linalg.norm(mesh.face_normals, axis=1) > .9)).any()
    assert blocks.slope_threshold == generator.slope_threshold == .75, "Block conversion must not mutate shared settings"
    terrains["blocks"].num_obstacles = 1
    assert light.scene.terrain.terrain_generator.sub_terrains["blocks"].num_obstacles == 300
    assert AntRandomizedEvalEnvCfg().rewards.to_dict() == AntEnvCfg().rewards.to_dict()
    print("[CHECK] One-stage baseline fine-tunes: no flat tiles, paired rewards, compatible PPO, 2 cm vertical blocks: PASS", flush=True)


def check_block_eval_configs():
    from isaaclab_tasks.manager_based.classic.ant_randomized.block_eval_env_cfg import AntBlockEvalEnvCfg
    from isaaclab_tasks.manager_based.classic.ant_randomized.baseline_finetune_env_cfg import baseline_finetune_terrain

    cfg = AntBlockEvalEnvCfg()
    cfg.validate()
    wave = AntRandomizedEvalEnvCfg()
    actual = cfg.to_dict()
    actual["scene"]["terrain"]["terrain_generator"] = wave.scene.terrain.terrain_generator.to_dict()
    assert actual == wave.to_dict(), "Only evaluation terrain may change"
    assert cfg.rewards.to_dict() == AntEnvCfg().rewards.to_dict()
    assert len(cfg.rewards.to_dict()) == 7 and cfg.scene.num_envs == 100
    assert cfg.events.push_robot is None
    terrain = cfg.scene.terrain.terrain_generator
    train = baseline_finetune_terrain()
    assert set(terrain.sub_terrains) == {"blocks"}
    assert terrain.seed == 2402 and terrain.seed != train.seed
    expected = train.sub_terrains["blocks"].to_dict()
    expected["proportion"] = 1.0
    assert terrain.sub_terrains["blocks"].to_dict() == expected
    terrain.sub_terrains["blocks"].num_obstacles = 1
    assert train.sub_terrains["blocks"].num_obstacles == 300
    assert AntBlockEvalEnvCfg().scene.terrain.terrain_generator.sub_terrains["blocks"].num_obstacles == 300
    loaded = load_cfg_from_registry("Isaac-Ant-DR-Eval-Blocks-v0", "env_cfg_entry_point")
    assert loaded.to_dict() == AntBlockEvalEnvCfg().to_dict()
    print("[CHECK] Block-only Eval: seven original rewards, unchanged dynamics, isolated geometry config: PASS", flush=True)


def check_physics():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = 24
    env = gym.make(args.task, cfg=cfg)
    try:
        obs, _ = env.reset()
        raw = env.unwrapped
        robot = raw.scene["robot"]
        assert torch.allclose(robot.data.default_root_state[:, 2], torch.full((args.num_envs,), .5, device=raw.device))
        assert obs["policy"].shape == (args.num_envs, 60)
        terrain = raw.scene.terrain
        assert not raw.scene.sensors, "DR must not add robot sensors"
        origins = raw.scene.env_origins.clone()
        original_grid = terrain._compute_env_origins_grid(args.num_envs, AntEnvCfg().scene.env_spacing)
        assert torch.equal(origins[:, :2], original_grid[:, :2]), "XY must match stock Ant placement"
        assert torch.unique(origins[:, :2], dim=0).shape[0] == args.num_envs, "No duplicated starts"
        assert torch.allclose(robot.data.root_pos_w[:, :2], origins[:, :2])
        assert torch.allclose(robot.data.root_pos_w[:, 2] - origins[:, 2], torch.full_like(origins[:, 2], .5))
        if args.num_envs == 100:
            for axis in (0, 1):
                coordinates = torch.unique(origins[:, axis], sorted=True)
                assert len(coordinates) == 10
                assert torch.allclose(coordinates.diff(), torch.full((9,), 5., device=raw.device))
        print(f"[CHECK] {args.num_envs} unique starts match the original 5 m grid; local spawn clearance 0.5 m: PASS", flush=True)
        materials = robot.root_physx_view.get_material_properties()
        assert torch.allclose(materials, materials[:, :1].expand_as(materials))
        assert (materials[..., 1] <= materials[..., 0]).all()
        friction = cfg.events.contact_friction.params
        if friction["num_buckets"] == 1:
            assert torch.allclose(materials[..., :2], torch.ones_like(materials[..., :2]))
        elif args.num_envs > 1:
            assert torch.unique(materials[:, 0, 0]).numel() > 1
        assert materials[..., 0].min() >= friction["static_friction_range"][0] - 1e-6
        assert materials[..., 0].max() <= friction["static_friction_range"][1] + 1e-6
        assert materials[..., 1].min() >= friction["dynamic_friction_range"][0] - 1e-6
        assert materials[..., 1].max() <= friction["dynamic_friction_range"][1] + 1e-6
        # Contact randomization must be reproducible even after unrelated RNG consumption.
        torch.rand(17)
        mdp.randomize_contact_friction(raw, None, **cfg.events.contact_friction.params)
        assert torch.equal(materials, robot.root_physx_view.get_material_properties())
        names = raw.reward_manager.active_terms
        stable_task = isinstance(cfg, AntRandomizedStableEnvCfg)
        forward_task = isinstance(cfg, AntRandomizedForwardEnvCfg)
        assert len(names) == (11 if forward_task else 9 if stable_task else 7)
        penalty_seen = torch.zeros(2, dtype=torch.bool, device=raw.device)
        forward_penalty_seen = torch.zeros(2, dtype=torch.bool, device=raw.device)
        for _ in range(args.steps):
            obs, reward, terminated, truncated, _ = env.step(torch.zeros((args.num_envs, 8), device=raw.device))
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
            weighted = raw.reward_manager._step_reward
            assert torch.allclose(reward, weighted.sum(dim=1) * raw.step_dt, atol=1e-6)
            if stable_task:
                # Auto-reset changes robot state, so compare live kinematics only on continuing episodes.
                continuing = ~(terminated | truncated)
                values = weighted[:, [names.index("vertical_velocity"), names.index("roll_pitch_velocity")]]
                assert (values <= 0).all(), "Stability terms must always be penalties"
                expected = torch.stack((
                    -0.5 * robot.data.root_lin_vel_w[:, 2].square(),
                    -0.05 * robot.data.root_ang_vel_b[:, :2].square().sum(dim=1),
                ), dim=1)
                assert torch.allclose(values[continuing], expected[continuing], atol=1e-5)
                penalty_seen |= (values < 0).any(dim=0)
            if forward_task:
                continuing = ~(terminated | truncated)
                values = weighted[:, [names.index("lateral_velocity"), names.index("heading_error")]]
                assert (values <= 0).all(), "Forward terms must be penalties"
                assert (values[:, 1] >= -2.0).all()
                # Independent horizontal-vector calculation checks the angle-based implementation.
                direction = math_utils.quat_apply(robot.data.root_quat_w, robot.data.FORWARD_VEC_B)[:, :2]
                goal = robot.data.root_pos_w.new_tensor(cfg.rewards.heading_error.params["target_pos"][:2])
                to_goal = goal - robot.data.root_pos_w[:, :2]
                cos_angle = (direction * to_goal).sum(dim=1) / (direction.norm(dim=1) * to_goal.norm(dim=1))
                expected = torch.stack((
                    cfg.rewards.lateral_velocity.weight * robot.data.root_lin_vel_b[:, 1].square(),
                    -(1.0 - cos_angle.clamp(-1.0, 1.0)),
                ), dim=1)
                assert torch.allclose(values[continuing], expected[continuing], atol=1e-5)
                forward_penalty_seen |= (values < 0).any(dim=0)
            if cfg.scene.terrain.terrain_type == "generator":
                ground_z = terrain.ground_height(robot.data.root_pos_w)
                assert torch.isfinite(ground_z).all(), "Ground must cover starts and this short rollout"
        env.reset()
        assert torch.equal(raw.scene.env_origins, origins), "Resets must retain fixed grid origins"
        assert torch.allclose(robot.data.root_pos_w[:, :2], origins[:, :2])
        if stable_task and args.steps > 0:
            assert penalty_seen.all(), "Both stability penalties must activate during physics stepping"
            print(
                "[CHECK] Both stability penalties are active, negative, and integrated with dt exactly once: PASS",
                flush=True,
            )
        if forward_task and args.steps > 0:
            assert forward_penalty_seen.all(), "Both forward penalties must activate during physics stepping"
            print("[CHECK] Body-lateral and horizontal heading penalties match live kinematics: PASS", flush=True)
        if cfg.events.push_robot is not None:
            before = robot.data.root_vel_w.clone()
            raw.event_manager.apply(mode="interval", dt=6.1)
            delta = robot.data.root_vel_w - before
            assert delta[:, :2].abs().max() > 0.0
            assert delta[:, :2].abs().max() <= 0.30001
            assert torch.allclose(delta[:, 2:], torch.zeros_like(delta[:, 2:]), atol=1e-6)
            print("[CHECK] Scheduled push affects only horizontal velocity within configured bounds: PASS", flush=True)
        print(
            f"[CHECK] {args.task}: {args.num_envs} environments, {args.steps} steps, friction and ground tracking: PASS",
            flush=True,
        )
    finally:
        env.close()


try:
    check_configs()
    check_finetune_configs()
    check_baseline_finetune_configs()
    check_block_eval_configs()
    check_physics()
finally:
    app.close()
