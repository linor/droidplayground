"""
walk_command.py -- commands for walking policies on the robot (robot_deploy.py).

A walking bundle's policy.meta.json carries a "command" block (written by
export_policy_for_deployment.py --gait-library): the command grid, the gait
period per grid point and the library's IK residual. The robot needs only that:
the policy observes (phase, command), and the phase advances by
control_dt / period(command) exactly like QminiLegEnv's walking mode.

CommandTable  -- trilinear period / IK-residual lookup, clamps a command into
                 the range the reference library could actually generate.
CommandInput  -- F710 sticks -> target command, ramped at the training rates.
"""
from __future__ import annotations

import math

import gamepad as gp

# Must match QminiLegEnvCfg.command_ramp_rate (qmini_leg_env.py): the command
# the policy sees ramps toward the stick at these rates (vx, vy, wz per s).
COMMAND_RAMP_RATE = (0.14, 0.09, 0.35)
# Must match QminiLegEnvCfg.command_max_ik_err: commands whose reference gait
# the library could not reach (joint limits at combined extremes) are scaled
# back toward zero until reachable -- training never sampled them.
COMMAND_MAX_IK_ERR = 0.005


class CommandTable:
    def __init__(self, meta_command: dict):
        self.axes = [list(map(float, meta_command[k])) for k in ("vx", "vy", "wz")]
        self.period = meta_command["period_s"]
        self.ik = meta_command.get("ik_err")
        self.lo = [a[0] for a in self.axes]
        self.hi = [a[-1] for a in self.axes]
        self.time_scale = float(meta_command.get("time_scale", 1.0))

    def _corners(self, cmd):
        idx, frac = [], []
        for v, ax in zip(cmd, self.axes):
            v = min(max(v, ax[0]), ax[-1])
            i = max(0, min(len(ax) - 2, next((k for k in range(len(ax) - 1) if v <= ax[k + 1]), len(ax) - 2)))
            idx.append(i); frac.append((v - ax[i]) / (ax[i + 1] - ax[i]))
        return idx, frac

    def _interp(self, table, cmd) -> float:
        (ix, iy, iz), (fx, fy, fz) = self._corners(cmd)
        out = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * (fz if dz else 1 - fz)
                    out += w * table[ix + dx][iy + dy][iz + dz]
        return out

    def period_s(self, cmd) -> float:
        return self._interp(self.period, cmd)

    def ik_err(self, cmd) -> float:
        return 0.0 if self.ik is None else self._interp(self.ik, cmd)

    def clamp(self, cmd):
        """Into the library box, then scaled toward zero until reachable."""
        c = [min(max(v, lo), hi) for v, lo, hi in zip(cmd, self.lo, self.hi)]
        if self.ik_err(c) <= COMMAND_MAX_IK_ERR:
            return c
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if self.ik_err([v * mid for v in c]) <= COMMAND_MAX_IK_ERR:
                lo = mid
            else:
                hi = mid
        return [v * lo for v in c]


class CommandInput:
    """Left stick: forward/back + strafe, right stick X: turn. Full deflection
    = the library maximum in that direction. The output command ramps toward
    the stick target at COMMAND_RAMP_RATE (what the policy was trained with)."""

    def __init__(self, table: CommandTable, pad: gp.Gamepad | None, dt: float):
        self.table, self.pad, self.dt = table, pad, dt
        self.cmd = [0.0, 0.0, 0.0]
        self.target = [0.0, 0.0, 0.0]
        self.hold_zero = False          # soft stop: ignore the sticks, ramp to zero
        self.was_connected = pad.connected if pad else False

    def update(self) -> list:
        connected = self.pad.poll() if self.pad else False
        if self.pad and connected and not self.hold_zero:
            lx, ly = gp.deadzone2(self.pad.axis(gp.AXIS_LX), self.pad.axis(gp.AXIS_LY), 0.12)
            rx = gp.deadzone(self.pad.axis(gp.AXIS_RX), 0.14)
            fwd, left, ccw = -ly, -lx, -rx          # stick up = forward, left = left, right stick left = CCW
            pick = lambda u, k: u * (self.table.hi[k] if u > 0 else -self.table.lo[k])
            self.target = self.table.clamp([pick(fwd, 0), pick(left, 1), pick(ccw, 2)])
        else:
            self.target = [0.0, 0.0, 0.0]
        for k, rate in enumerate(COMMAND_RAMP_RATE):
            d = self.target[k] - self.cmd[k]
            m = rate * self.dt
            self.cmd[k] += d if abs(d) <= m else math.copysign(m, d)
        self.was_connected = connected
        return list(self.cmd)

    def is_zero(self, eps: float = 1e-4) -> bool:
        return all(abs(v) < eps for v in self.cmd)
