from isaaclab.managers import SceneEntityCfg
import torch

def target_angle(env):
    return env.command_manager.get_command("target_angle")


def target_angle_error(env, asset_cfg: SceneEntityCfg):
    robot = env.scene[asset_cfg.name]
    target = env.command_manager.get_command("target_angle")
    current = robot.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.atan2(torch.sin(target - current), torch.cos(target - current))