# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def track_target_angle_exp(env, asset_cfg: SceneEntityCfg, std: float = 0.25):
    robot = env.scene[asset_cfg.name]
    target = env.command_manager.get_command("target_angle")
    current = robot.data.joint_pos[:, asset_cfg.joint_ids]

    error = torch.atan2(torch.sin(target - current), torch.cos(target - current))
    return torch.exp(-torch.square(error[:, 0]) / std**2)


def joint_vel_l2_penalty(env, asset_cfg: SceneEntityCfg):
    robot = env.scene[asset_cfg.name]
    return torch.square(robot.data.joint_vel[:, asset_cfg.joint_ids]).sum(dim=1)