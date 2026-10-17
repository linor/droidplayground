"""
gen_gait_library.py -- command-indexed reference gait library for the walking env.

Uses the stylized_gait generator in ../../qmini-gait (Hopkins et al. 2024): for
every command (vx, vy, wz) on a grid, the paper's Sec. 5.3 interpolation weights
(ported from qmini-gait/web/param_step.js blendWeights) pick a convex
combination of the 9 key gait samples, the PARAMETERS are blended (Sec. 5.2,
never poses) and the planner solves one periodic cycle with leg IK. The env
interpolates between neighbouring grid cycles at the same normalized phase.

Conventions, all verified 2026-10-12:
  * joint signs: solve_sample()'s skeleton angles times the physics-graph sign
    map of qmini-gait/scripts/export_isaac_motion.py reproduce the hand-converted
    step-in-place keyframes (left yaw/roll/pitch/knee and right yaw/roll/ankle
    negated) to 0.00 deg, and our fk_geometry.json FK on the exported angles
    reproduces the generator's foot yaw and foot path to 0.00 on the turn,
    strafe, forward and reverse gaits (so hip yaw is right too).
  * stance offset: the same constant joint offset as the step-in-place
    keyframes (left_pitch -16, left_knee +10, left_ankle +1, mirrored), so the
    policy's default pose / deploy config stay unchanged.
  * time scale: joint motion is played TIME_SCALE x slower than authored (the
    step-in-place policies ran at 2x); commanded velocities scale by
    1/TIME_SCALE accordingly. The library stores velocities as the robot will
    actually move.
  * phase: normalized cycle time tau = t / T in [0, 1). Contact events sit at
    nearly the same tau for every sample (lift-off 0 / 0.5, touch-down
    0.34-0.36 / 0.84-0.86), so equal-tau interpolation between neighbouring
    grid cycles is close to the paper's registered blend.
  * smoothing: every generated gait has a velocity kink right at swing onset
    (up to 5.9 deg/frame at 60 fps, the same C1 discontinuity that was
    hand-smoothed out of the step-in-place keyframes on 2026-09-03 after it
    caused real-robot runaways). A periodic Gaussian low-pass in phase removes
    it for every cycle.

usage: python3 gen_gait_library.py [--out PATH] [--time-scale 2.0] [--nph 100]
"""

import argparse, json, math, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # one thread per worker: BLAS threading made each solve ~5x slower
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "..", "..", "qmini-gait")
sys.path[:0] = [os.path.join(GEN, "src"), os.path.join(GEN, "scripts")]
from stylized_gait import samples as SM, overrides as GO, paths as GP          # noqa: E402
from stylized_gait.core import f_soft, SOLE_CORNERS                            # noqa: E402
from stylized_gait.blend import BlendedSample                                  # noqa: E402
from stylized_gait.planner import solve_sample                                 # noqa: E402
import export_isaac_motion as EX                                               # noqa: E402

JOINT_ORDER = ["left_yaw", "left_roll", "left_pitch", "left_knee", "left_ankle",
               "right_yaw", "right_roll", "right_pitch", "right_knee", "right_ankle"]
STANCE_OFFSET_DEG = {"left_pitch": -16.0, "left_knee": 10.0, "left_ankle": 1.0,
                     "right_pitch": 16.0, "right_knee": -10.0, "right_ankle": -1.0}
LIMIT_MARGIN_DEG = 1.0          # robot_deploy.py / env target clamp margin


def blend_weights(g, v, eta=0.5):
    """Port of web/param_step.js blendWeights (paper Sec. 5.3, Eqs 8-10):
    {sample_name: weight} for command v = (vx, vy, wz) in generator units."""
    fr = ["reverse_fast", "reverse_slow", "step_in_place", "forward_slow", "forward_fast"]
    frv = [g[n].vel[0] for n in fr]
    ts = ["left_turn", "left_strafe", "right_turn", "right_strafe"]
    tsb = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
    w = {n: 0.0 for n in g}
    x = min(max(v[0], frv[0]), frv[-1])
    i = 0
    while i < len(frv) - 2 and x > frv[i + 1]:
        i += 1
    a_fr = f_soft((x - frv[i]) / (frv[i + 1] - frv[i]))
    yn, tn = v[1] / SM.Y_MAX, v[2] / SM.TH_MAX
    rho = min(math.hypot(yn, tn), 1.0)
    beta = math.atan2(yn, tn)
    if beta < 0:
        beta += 2 * math.pi
    seg = min(3, int(beta // (math.pi / 2)))
    a_b = f_soft((beta - tsb[seg]) / (math.pi / 2))
    xn = abs(v[0]) / SM.X_MAX
    a_ts = f_soft(rho) * (1 - eta * f_soft(xn))
    w[fr[i]] += (1 - a_ts) * (1 - a_fr)
    w[fr[i + 1]] += (1 - a_ts) * a_fr
    w[ts[seg]] += a_ts * (1 - a_b)
    w[ts[(seg + 1) % 4]] += a_ts * a_b
    return {k: float(val) for k, val in w.items() if val > 1e-4}


def periodic_gauss(x, sigma):
    """Periodic Gaussian low-pass along axis 0 (sigma in samples)."""
    if sigma <= 0:
        return x
    n = x.shape[0]
    k = np.arange(-n // 2, n // 2)
    ker = np.exp(-0.5 * (k / sigma) ** 2); ker /= ker.sum()
    X = np.fft.rfft(x, axis=0); K = np.fft.rfft(np.fft.ifftshift(ker))
    return np.fft.irfft(X * K[:, None], n=n, axis=0)


def solve_cycle(g, cmd, nph, signs):
    """One cycle for generator-unit command cmd -> (T, joint_deg (nph,10) in
    JOINT_ORDER/PhysX sign without offset, foot_h (nph,2), contact (nph,2))."""
    ws = blend_weights(g, cmd)
    names = list(ws)
    s = BlendedSample([g[n] for n in names], [ws[n] for n in names], vel=np.asarray(cmd, float))
    r = solve_sample(s, fps=nph / s.T, passes=3)
    assert r["n"] == nph, (r["n"], nph)
    q = np.concatenate([r["q"]["L"], r["q"]["R"]], axis=1) * signs
    fh = np.zeros((nph, 2)); ct = np.zeros((nph, 2))
    for k, side in enumerate("LR"):
        for i in range(nph):
            Rf = r["foot_R"][side][i]
            fh[i, k] = min(r["foot_p"][side][i][2] + (Rf @ c)[2] for c in SOLE_CORNERS[side])
        ct[:, k] = r["contact"][side]
    ik_err = max(float(np.abs(r["err"][sd]).max()) for sd in "LR")
    return s.T, np.degrees(q), fh, ct, ws, ik_err


_G = _SIGNS = None


def _set_pelvis_style(sway, roll_scale):
    # module-level in stylized_gait.samples so overrides.apply() (which re-runs
    # build()) sees the same values; see samples.PELVIS_SWAY
    SM.PELVIS_SWAY, SM.PELVIS_ROLL_SCALE = float(sway), float(roll_scale)


def _worker_init(sway=0.0, roll_scale=1.0):
    global _G, _SIGNS
    _set_pelvis_style(sway, roll_scale)
    _G = GO.apply(SM.build(), GP.OVERRIDES_JSON if os.path.isfile(GP.OVERRIDES_JSON) else None)
    sg, _ = EX.joint_signs_and_limits()
    _SIGNS = np.array([sg[EX.SKEL2PHYS[nm]] for nm in EX.ORDER], float)


def _solve(args):
    key, cmd, nph = args
    return key, cmd, solve_cycle(_G, cmd, nph, _SIGNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "rl", "source", "droidplayground", "droidplayground",
                                                  "tasks", "direct", "droidplayground", "gait_library_2x.npz"))
    ap.add_argument("--time-scale", type=float, default=2.0)
    ap.add_argument("--nph", type=int, default=100, help="samples per cycle (uniform normalized phase)")
    ap.add_argument("--smooth-sigma", type=float, default=2.0, help="periodic Gaussian sigma, in phase samples")
    ap.add_argument("--vx", default="-0.35,-0.25,-0.15,-0.075,0,0.075,0.15,0.25,0.35", help="generator units (m/s at 1x)")
    ap.add_argument("--vy", default="-0.16,-0.08,0,0.08,0.16")
    ap.add_argument("--wz", default="-0.6,-0.3,0,0.3,0.6")
    ap.add_argument("--check", type=int, default=6, help="random off-grid commands to check interpolation against")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--extra-offset", default="",
                    help="additional stance offset per joint type in the LEFT leg's (PhysX) convention, mirrored onto the "
                         "right leg (negated), e.g. 'pitch=-7.2,knee=-5.5,ankle=-7.9,yaw=-4.0,roll=-0.55'")
    ap.add_argument("--pelvis-sway", type=float, default=0.0,
                    help="lateral pelvis translation toward the support foot, m (stylized_gait samples.PELVIS_SWAY); "
                         "0 = the authored gaits")
    ap.add_argument("--pelvis-roll-scale", type=float, default=1.0,
                    help="scale on the authored pelvis lean toward the support foot (samples.PELVIS_ROLL_SCALE)")
    a = ap.parse_args()
    _set_pelvis_style(a.pelvis_sway, a.pelvis_roll_scale)

    g = GO.apply(SM.build(), GP.OVERRIDES_JSON if os.path.isfile(GP.OVERRIDES_JSON) else None)
    sg, limits = EX.joint_signs_and_limits()
    signs = np.array([sg[EX.SKEL2PHYS[nm]] for nm in EX.ORDER], float)
    assert [EX.SKEL2PHYS[nm].removeprefix("Revolute_") for nm in EX.ORDER] == JOINT_ORDER
    lo = np.array([math.degrees(limits["Revolute_" + j][0]) for j in JOINT_ORDER]) + LIMIT_MARGIN_DEG
    hi = np.array([math.degrees(limits["Revolute_" + j][1]) for j in JOINT_ORDER]) - LIMIT_MARGIN_DEG
    extra = {k: float(v) for k, v in (kv.split("=") for kv in a.extra_offset.split(",") if kv)}
    off = np.array([STANCE_OFFSET_DEG.get(j, 0.0) + (1.0 if j.startswith("left_") else -1.0) * extra.get(j.split("_", 1)[1], 0.0)
                    for j in JOINT_ORDER])
    print("stance offset (deg):", dict(zip(JOINT_ORDER, np.round(off, 2))))

    VX, VY, WZ = (np.array([float(v) for v in s.split(",")]) for s in (a.vx, a.vy, a.wz))
    shape = (len(VX), len(VY), len(WZ))
    period = np.zeros(shape); joints = np.zeros(shape + (a.nph, 10)); foot_h = np.zeros(shape + (a.nph, 2))
    contact = np.zeros(shape + (a.nph, 2)); ik_err = np.zeros(shape)
    raw_jump = sm_jump = 0.0; worst_ik = 0.0; clipped = 0
    t0 = time.time()

    def finish(q):
        nonlocal raw_jump, sm_jump, clipped
        def acc(x):   # largest change of per-sample velocity (periodic), deg
            v = np.roll(x, -1, axis=0) - x
            return float(np.abs(np.roll(v, -1, axis=0) - v).max())
        raw_jump = max(raw_jump, acc(q))
        qs = periodic_gauss(q, a.smooth_sigma) + off
        sm_jump = max(sm_jump, acc(qs - off))
        c = np.clip(qs, lo, hi); clipped += int((c != qs).any(axis=1).sum()); return c

    rng = np.random.default_rng(0)
    checks = [(rng.uniform(VX[0], VX[-1]), rng.uniform(VY[0], VY[-1]), rng.uniform(WZ[0], WZ[-1])) for _ in range(a.check)]
    jobs = [((ix, iy, iz), (vx, vy, wz), a.nph) for ix, vx in enumerate(VX) for iy, vy in enumerate(VY) for iz, wz in enumerate(WZ)]
    jobs += [(("check", k), c, a.nph) for k, c in enumerate(checks)]
    check_res = {}; worst_ik_cmd = None
    with mp.Pool(a.workers, initializer=_worker_init, initargs=(a.pelvis_sway, a.pelvis_roll_scale)) as pool:
        for done, (key, cmd, (T, q, fh, ct, ws, ik)) in enumerate(pool.imap_unordered(_solve, jobs), 1):
            if ik > worst_ik:
                worst_ik, worst_ik_cmd = ik, cmd
            if key[0] == "check":
                check_res[key[1]] = (T, finish(q)); continue
            period[key] = T * a.time_scale; ik_err[key] = ik
            joints[key] = finish(q); foot_h[key] = fh; contact[key] = ct
            if done % 20 == 0:
                print(f"  {done}/{len(jobs)} cycles ({time.time() - t0:.0f} s)", flush=True)

    # interpolation check: off-grid commands, trilinear in command at equal phase vs a direct solve
    errs = []
    for k, cmd in enumerate(checks):
        T, q = check_res[k]
        idx = []; fr = []
        for v, ax in zip(cmd, (VX, VY, WZ)):
            i = int(np.clip(np.searchsorted(ax, v) - 1, 0, len(ax) - 2)); idx.append(i); fr.append((v - ax[i]) / (ax[i + 1] - ax[i]))
        qi = np.zeros_like(q); Ti = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (fr[0] if dx else 1 - fr[0]) * (fr[1] if dy else 1 - fr[1]) * (fr[2] if dz else 1 - fr[2])
                    qi += w * joints[idx[0] + dx, idx[1] + dy, idx[2] + dz]; Ti += w * period[idx[0] + dx, idx[1] + dy, idx[2] + dz]
        errs.append((cmd, float(np.abs(qi - q).max()), Ti, T * a.time_scale))
    del g, signs

    ts = a.time_scale
    np.savez_compressed(
        a.out,
        joint_order=np.array(JOINT_ORDER), vx=VX / ts, vy=VY / ts, wz=WZ / ts,
        period_s=period.astype(np.float32), joint_pos_deg=joints.astype(np.float32),
        foot_height_m=foot_h.astype(np.float32), contact=contact.astype(np.float32),
        ik_err=ik_err.astype(np.float32),
        time_scale=np.float32(ts), stance_offset_deg=off.astype(np.float32),
        info=np.array(json.dumps({
            "generator": "qmini-gait stylized_gait (Hopkins et al. 2024), Sec. 5.3 weights + Sec. 5.2 parameter blend",
            "phase": "normalized cycle time tau = t/T, uniform, nph samples; left lift-off at 0, right at 0.5",
            "velocities": "vx forward, vy left, wz CCW, in robot units AFTER the time scale",
            "joint_pos_deg": "PhysX sign convention, JOINT_ORDER, stance offset applied, clipped to limits -1 deg",
            "foot_height_m": "lowest sole corner above the ground per foot (L, R), generator geometry, no offset",
            "ik_err": "worst leg-IK residual over the cycle (m or rad); > ~0.005 = target unreachable (joint limits)",
            "smooth_sigma_samples": a.smooth_sigma,
            "pelvis_sway_m": a.pelvis_sway, "pelvis_roll_scale": a.pelvis_roll_scale, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        })),
    )
    print(f"\nwrote {a.out}  grid {shape} x {a.nph} phase samples, time scale {ts}")
    print(f"  robot command ranges: vx {VX[0] / ts:+.3f}..{VX[-1] / ts:+.3f} m/s, vy {VY[0] / ts:+.3f}..{VY[-1] / ts:+.3f} m/s, wz {WZ[0] / ts:+.2f}..{WZ[-1] / ts:+.2f} rad/s")
    print(f"  periods {period.min():.2f}..{period.max():.2f} s; worst IK residual {worst_ik:.2e} at generator command {tuple(round(c, 3) for c in worst_ik_cmd)}")
    print(f"  swing-onset kink (max velocity change per sample): raw {raw_jump:.2f} deg -> smoothed {sm_jump:.2f} deg; limit-clipped samples: {clipped}")
    for cmd, e, Ti, T in errs:
        print(f"  interpolation check cmd ({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}): max joint error {e:.2f} deg, period {Ti:.3f} vs {T:.3f} s")


if __name__ == "__main__":
    main()
