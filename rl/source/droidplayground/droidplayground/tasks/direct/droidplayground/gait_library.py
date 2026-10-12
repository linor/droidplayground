"""Command-indexed reference gait library for the walking task.

Built offline by experiments/qmini-leg/gen_gait_library.py from the stylized_gait
generator (qmini-gait/): one periodic cycle per (vx, vy, wz) grid point, sampled
at a uniform normalized phase tau = t / T in [0, 1). At run time every env gets
its reference by trilinear interpolation over its current command and linear
interpolation in phase -- between neighbouring grid cycles this stays within
~0.3-1.2 deg of a direct solve (gen_gait_library.py prints that check).

Two small adapters let QminiLegEnv keep calling `.sample(motion_time)` /
`.length` exactly as it does for the keyframe MotionPlayer: in walking mode
motion_time IS the phase (length 1.0) and advances by step_dt / period(cmd).
"""

from __future__ import annotations

import numpy as np
import torch


class GaitLibrary:
    def __init__(self, path, joint_names: list[str], device):
        z = np.load(path, allow_pickle=False)
        self.device = device
        order = [str(n) for n in z["joint_order"]]
        names = [n.removeprefix("Revolute_") for n in joint_names]
        missing = [n for n in names if n not in order]
        if missing:
            raise ValueError(f"gait library {path} has no column for {missing} (it has {order})")
        cols = [order.index(n) for n in names]
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
        self.axes = [t(z["vx"]), t(z["vy"]), t(z["wz"])]
        self.shape = tuple(len(a) for a in self.axes)
        self.nph = int(z["joint_pos_deg"].shape[3])
        self.period = t(z["period_s"])                                          # (nx, ny, nz)
        self.joints = torch.deg2rad(t(z["joint_pos_deg"][..., cols]))           # (nx, ny, nz, nph, nj)
        fh = z["foot_height_m"]                                                 # (..., nph, 2) L, R
        # env order for foot heights: [left_heel, left_toe, right_heel, right_toe]
        self.foot = t(np.stack([fh[..., 0], fh[..., 0], fh[..., 1], fh[..., 1]], axis=-1))
        self.ik_err = t(z["ik_err"]) if "ik_err" in z.files else torch.zeros(self.shape, device=device)
        self.time_scale = float(z["time_scale"])
        self.lo = torch.stack([a[0] for a in self.axes]); self.hi = torch.stack([a[-1] for a in self.axes])

    # -- command interpolation -------------------------------------------------
    def _corners(self, cmd: torch.Tensor):
        """8 trilinear corners: flat grid index (N, 8) and weights (N, 8)."""
        idx, frac = [], []
        for d, ax in enumerate(self.axes):
            v = cmd[:, d].clamp(ax[0], ax[-1])
            i = torch.clamp(torch.searchsorted(ax, v.contiguous()) - 1, 0, len(ax) - 2)
            idx.append(i); frac.append((v - ax[i]) / (ax[i + 1] - ax[i]))
        _, ny, nz = self.shape
        flat, w = [], []
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    flat.append(((idx[0] + dx) * ny + (idx[1] + dy)) * nz + (idx[2] + dz))
                    w.append((frac[0] if dx else 1 - frac[0]) * (frac[1] if dy else 1 - frac[1]) * (frac[2] if dz else 1 - frac[2]))
        return torch.stack(flat, 1), torch.stack(w, 1)

    def _interp_cmd(self, table: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        flat, w = self._corners(cmd)
        vals = table.reshape(-1, *table.shape[3:])[flat]                        # (N, 8, ...)
        return (vals * w.view(*w.shape, *([1] * (vals.dim() - 2)))).sum(1)

    def period_s(self, cmd):
        return self._interp_cmd(self.period, cmd)

    def ik_error(self, cmd):
        return self._interp_cmd(self.ik_err, cmd)

    def _interp_phase(self, table, cmd, phase):
        """table (nx, ny, nz, nph, k) -> (N, k) at each env's command and phase."""
        flat, w = self._corners(cmd)
        x = (phase % 1.0) * self.nph
        i0 = torch.floor(x).long() % self.nph; i1 = (i0 + 1) % self.nph; a = (x - torch.floor(x)).unsqueeze(-1)
        tab = table.reshape(-1, self.nph, table.shape[-1])                      # (G, nph, k)
        v0 = tab[flat, i0.unsqueeze(1).expand_as(flat)]                         # (N, 8, k)
        v1 = tab[flat, i1.unsqueeze(1).expand_as(flat)]
        v = v0 + a.unsqueeze(1) * (v1 - v0)
        return (v * w.unsqueeze(-1)).sum(1)

    def joint_ref(self, cmd, phase):
        return self._interp_phase(self.joints, cmd, phase)

    def foot_ref(self, cmd, phase):
        return self._interp_phase(self.foot, cmd, phase)


class LibraryMotion:
    """MotionPlayer-compatible view: sample(phase) -> joint reference for the
    env's CURRENT commands; length 1.0 (motion_time is the phase)."""

    length = 1.0

    def __init__(self, lib: GaitLibrary, get_cmd, kind="joints"):
        self.lib, self.get_cmd, self.kind = lib, get_cmd, kind

    def sample(self, t):
        if self.kind == "joints":
            return self.lib.joint_ref(self.get_cmd(), t)
        return self.lib.foot_ref(self.get_cmd(), t)
