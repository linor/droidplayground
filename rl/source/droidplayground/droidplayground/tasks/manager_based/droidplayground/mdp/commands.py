import torch
from isaaclab.managers import CommandTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaacsim.core.utils.torch.rotations import quat_from_euler_xyz
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply

@configclass
class RandomJointAngleCommandCfg(CommandTermCfg):
    class_type: type = None
    angle_range: tuple[float, float] = (-torch.pi, torch.pi)
    
    asset_name: str = "robot"
    joint_name: str = "Revolute_1"
    body_name: str = "horn_1"
    
    debug_vis: bool = True
    debug_env_count: int = 1
    line_length: float = 0.04
    line_z_offset: float = 0.02
    line_width: float = 3.0
    joint_center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    visual_angle_offset: float = 0.0


class RandomJointAngleCommand(CommandTerm):
    cfg: RandomJointAngleCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        
        self.robot = env.scene[cfg.asset_name]
        self.joint_id = self.robot.find_joints(cfg.joint_name)[0][0]

        self.target_angle = torch.zeros(
            self.num_envs, 1, device=self.device
        )
 
        self._debug_draw = None
        if cfg.debug_vis:
            try:
                from isaacsim.util.debug_draw import _debug_draw
                self._debug_draw = _debug_draw.acquire_debug_draw_interface()
            except ImportError:
                self._debug_draw = None

            self.body_id = self.robot.find_bodies(cfg.body_name)[0][0]

    def _draw_debug_lines(self):
        if self._debug_draw is None:
            return

        self._debug_draw.clear_lines()

        n = min(self.cfg.debug_env_count, self.num_envs)

        body_pos_w = self.robot.data.body_pos_w[:n, self.body_id]
        body_quat_w = self.robot.data.body_quat_w[:n, self.body_id]

        current_angle = self.robot.data.joint_pos[:n, self.joint_id] + self.cfg.visual_angle_offset
        target_angle = self.target_angle[:n, 0] + self.cfg.visual_angle_offset

        joint_offset_b = torch.tensor(
            self.cfg.joint_center_offset,
            device=self.device,
            dtype=body_pos_w.dtype,
        ).unsqueeze(0).repeat(n, 1)

        joint_center_w = body_pos_w + quat_apply(body_quat_w, joint_offset_b)
        joint_center_w[:, 2] += self.cfg.line_z_offset

        length = self.cfg.line_length

        current_ends = joint_center_w.clone()
        current_ends[:, 0] += length * torch.cos(current_angle)
        current_ends[:, 1] += length * torch.sin(current_angle)

        target_ends = joint_center_w.clone()
        target_ends[:, 0] += length * torch.cos(target_angle)
        target_ends[:, 1] += length * torch.sin(target_angle)

        starts = []
        ends = []
        colors = []
        widths = []

        for i in range(n):
            start = tuple(joint_center_w[i].detach().cpu().tolist())

            starts.append(start)
            ends.append(tuple(current_ends[i].detach().cpu().tolist()))
            colors.append((0.0, 1.0, 0.0, 1.0))  # current = green
            widths.append(self.cfg.line_width)

            starts.append(start)
            ends.append(tuple(target_ends[i].detach().cpu().tolist()))
            colors.append((0.0, 0.0, 1.0, 1.0))  # target = blue
            widths.append(self.cfg.line_width)

        self._debug_draw.draw_lines(starts, ends, colors, widths)
        
    @property
    def command(self) -> torch.Tensor:
        return self.target_angle

    def _resample_command(self, env_ids: torch.Tensor):
        low, high = self.cfg.angle_range
        self.target_angle[env_ids, 0] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(low, high)
        
    def _update_command(self):
        if self.cfg.debug_vis:
            self._draw_debug_lines()
            
    def _update_metrics(self):
        pass
    
RandomJointAngleCommandCfg.class_type = RandomJointAngleCommand