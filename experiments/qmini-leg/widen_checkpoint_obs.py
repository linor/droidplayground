"""
widen_checkpoint_obs.py -- warm-start an rsl_rl checkpoint on a wider observation.

For the walking task: the step-in-place policies observe 28 values, the walking
task appends the command (vx, vy, wz) -> 31. Every tensor whose input side is
the observation (actor.0 / critic.0 weights, both observation normalizers and
the matching Adam moments) is padded:
  * first-layer weights: zero columns, so the widened network computes EXACTLY
    the old function at first (it steps in place whatever the command) and
    learns the command dependence from there;
  * normalizers: mean 0 and the expected std of the new inputs (--new-std), so
    the normalized command is O(1) from the first update;
  * Adam moments: zeros.

usage: python3 widen_checkpoint_obs.py SRC.pt DST.pt [--new-std 0.074,0.033,0.123]
(the default stds: uniform commands within the 2x library range, 20 % zeros)
"""
import argparse, os
import torch

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--old-dim", type=int, default=28)
ap.add_argument("--new-std", default="0.074,0.033,0.123")
a = ap.parse_args()
new_std = torch.tensor([float(v) for v in a.new_std.split(",")])
n_new = len(new_std); old = a.old_dim
ck = torch.load(a.src, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
for k, v in list(sd.items()):
    if k.endswith(("actor.0.weight", "critic.0.weight")):
        assert v.shape[1] == old, (k, v.shape)
        sd[k] = torch.cat([v, torch.zeros(v.shape[0], n_new, dtype=v.dtype)], 1)
    elif "obs_normalizer" in k and k.rsplit(".", 1)[1] in ("_mean", "_var", "_std"):
        assert v.shape[-1] == old, (k, v.shape)
        kind = k.rsplit(".", 1)[1]
        pad = torch.zeros(n_new) if kind == "_mean" else (new_std ** 2 if kind == "_var" else new_std.clone())
        sd[k] = torch.cat([v, pad.view(1, -1).to(v.dtype)], 1)
    else:
        continue
    print(f"  {k}: {tuple(v.shape)} -> {tuple(sd[k].shape)}")
n_opt = 0
for st in ck.get("optimizer_state_dict", {}).get("state", {}).values():
    for name in ("exp_avg", "exp_avg_sq"):
        t = st.get(name)
        if t is not None and t.dim() == 2 and t.shape[1] == old:
            st[name] = torch.cat([t, torch.zeros(t.shape[0], n_new, dtype=t.dtype)], 1); n_opt += 1
print(f"  optimizer moments padded: {n_opt}")
os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
torch.save(ck, a.dst)
print(f"wrote {a.dst} (iteration {ck.get('iter')})")
