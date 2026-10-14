"""Left/right mirror symmetry for the walking task (rsl_rl symmetry_cfg).

Added 2026-10-14: with a mirror-symmetric reference (gait library v2) the
stage-2 walking policy still held the LEFT leg in a permanently different pose
(hip pitch -15.5 deg and knee -11 deg off the reference, the left foot standing
on its toes with the heel 5 cm up, while the right leg tracked the reference
with a flat foot). The robot geometry, joint limits and foot soles are mirror
images, so this is an asymmetric habit of the policy -- a limp -- which nothing
in PPO discourages. The mirror loss penalizes the policy for acting
differently in the mirrored situation (Mittal et al. 2024, rsl_rl symmetry).

Mirroring = reflection across the sagittal plane (robot base axes: +X left,
+Y rear, +Z up), plus a half-cycle phase shift (the left leg at phase p mirrors
the right leg at p + 0.5):
  joint pos / vel / residual actions (PhysX sign convention): swap left and
      right and negate -- the convention's own antisymmetry, verified on the
      library to 0.10 deg: q_R(cmd*, p + 0.5) = -q_L(cmd, p)
  projected gravity (a vector):          x -> -x
  angular velocity (a pseudovector):     (wx, wy, wz) -> (wx, -wy, -wz)
  phase sin / cos (shift by half a cycle): both negated
  command (vx fwd, vy left, wz ccw):     (vx, -vy, -wz)
Observation layout = QminiLegEnv._get_observations in walking mode:
  joint_pos(n) joint_vel(n) gravity(3) ang_vel(3) sin cos cmd(3).
"""
from __future__ import annotations

import torch

_PERM_CACHE: dict = {}


def _joint_perm(env) -> torch.Tensor:
    key = id(env)
    if key not in _PERM_CACHE:
        u = env.unwrapped
        names = [n.removeprefix("Revolute_") for n in u.robot.joint_names[: u.num_joints]]
        other = lambda n: ("right_" + n[5:]) if n.startswith("left_") else ("left_" + n[6:])
        _PERM_CACHE[key] = torch.tensor([names.index(other(n)) for n in names], device=u.device)
    return _PERM_CACHE[key]


def _mirror_joints(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return -x[..., perm]


def _mirror_obs(o: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    n = perm.numel()
    m = o.clone()
    m[:, :n] = _mirror_joints(o[:, :n], perm)
    m[:, n:2 * n] = _mirror_joints(o[:, n:2 * n], perm)
    g = 2 * n
    m[:, g] = -o[:, g]                       # gravity x
    m[:, g + 4] = -o[:, g + 4]               # ang vel y
    m[:, g + 5] = -o[:, g + 5]               # ang vel z
    m[:, g + 6:g + 8] = -o[:, g + 6:g + 8]   # phase sin, cos (+half cycle)
    if o.shape[1] >= g + 11:                 # walking: command
        m[:, g + 9] = -o[:, g + 9]           # vy
        m[:, g + 10] = -o[:, g + 10]         # wz
    return m


def qmini_mirror(env, obs=None, actions=None):
    """rsl_rl data_augmentation_func: returns [original; mirrored] along the batch."""
    perm = _joint_perm(env)
    obs_out = act_out = None
    if obs is not None:
        mirrored = obs.clone()
        for k in obs.keys():
            mirrored[k] = _mirror_obs(obs[k], perm)
        obs_out = torch.cat([obs, mirrored], dim=0)
    if actions is not None:
        act_out = torch.cat([actions, _mirror_joints(actions, perm)], dim=0)
    return obs_out, act_out
