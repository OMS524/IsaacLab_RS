"""Ant terrain resets, contact randomization, termination, and stability rewards."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_contact_friction(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    num_buckets: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Assign one contact material per Ant, using the same value on every collision shape.

    Applied only at startup. The ground keeps its original unit friction and
    average combination; robot-side materials still randomize contact friction.
    Geometry, mass, inertia, and the robot USD are never modified.
    A private CPU RNG keeps the friction draw independent of push-event scheduling.
    """
    asset = env.scene[asset_cfg.name]
    materials = asset.root_physx_view.get_material_properties()
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    generator = torch.Generator(device="cpu").manual_seed(int(env.cfg.seed) + 1701)
    samples = torch.rand((num_buckets, 2), generator=generator)
    static = static_friction_range[0] + samples[:, 0] * (static_friction_range[1] - static_friction_range[0])
    dynamic = dynamic_friction_range[0] + samples[:, 1] * (dynamic_friction_range[1] - dynamic_friction_range[0])
    buckets = torch.stack((static, torch.minimum(static, dynamic), torch.zeros_like(static)), dim=-1)
    # Draw for all environments before subselecting, so the assignment is stable by env ID.
    bucket_ids = torch.randint(num_buckets, (env.num_envs,), generator=generator)
    materials[env_ids] = buckets[bucket_ids[env_ids]].unsqueeze(1)
    asset.root_physx_view.set_material_properties(materials, env_ids)


def root_height_below_terrain(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Use clearance above the ground instead of absolute world z to detect falls.

    Query the static terrain directly, without adding a robot sensor or policy input.
    Outside the finite mesh, use the original world-height fall criterion.
    A missing ray hit alone does not terminate an episode.
    """
    root_pos = env.scene[asset_cfg.name].data.root_pos_w
    ground_z = env.scene.terrain.ground_height(root_pos)
    root_z = root_pos[:, 2]
    clearance = torch.where(torch.isfinite(ground_z), root_z - ground_z, root_z)
    return clearance < minimum_height


def world_lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared vertical torso velocity in the world frame (z is gravity-opposed).

    Unlike body-frame z velocity, this does not project horizontal translation
    onto the penalized axis when the torso pitches or rolls. A negative reward
    weight discourages excessive bouncing while still permitting terrain following.
    """
    return torch.square(env.scene[asset_cfg.name].data.root_lin_vel_w[:, 2])


def body_lin_vel_y_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared sideways torso velocity in the body frame, not the world y axis."""
    return torch.square(env.scene[asset_cfg.name].data.root_lin_vel_b[:, 1])


def heading_error_cost(
    env: ManagerBasedRLEnv,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return 1 - cos(theta) for horizontal heading error toward the Ant goal.

    The body's +x axis defines its heading. Cosine makes the cost continuous
    across the +/-pi boundary and penalizes errors even inside the original bonus plateau.
    The cost lies in [0, 2]; apply a negative reward weight to penalize misalignment.
    When already at the goal in xy, no horizontal target direction is defined,
    so this term returns zero. Height differences do not change the goal bearing.
    """
    asset = env.scene[asset_cfg.name]
    to_target_xy = asset.data.root_pos_w.new_tensor(target_pos[:2]) - asset.data.root_pos_w[:, :2]
    target_heading = torch.atan2(to_target_xy[:, 1], to_target_xy[:, 0])
    cost = 1.0 - torch.cos(target_heading - asset.data.heading_w)
    return torch.where(to_target_xy.square().sum(dim=1) > 1.0e-12, cost, torch.zeros_like(cost))
