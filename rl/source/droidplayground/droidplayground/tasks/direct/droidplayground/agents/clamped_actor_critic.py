# Subclasses rsl_rl's ActorCritic purely to put a hard ceiling on the
# policy's exploration std, which the library leaves completely unbounded
# (self.std is a plain nn.Parameter, see rsl_rl/modules/actor_critic.py --
# no clamping anywhere in that file, confirmed by reading it directly).
#
# Added 2026-09-06 after TWO independent catastrophic divergences in this
# project both showed the same signature: Mean action noise std climbing
# without limit (0.81, then 8.9, then 4.4+ and still rising at the final
# iteration each time) while mean_reward went negative and tracking errors
# blew up. Root cause isn't just "unbounded parameter" in isolation -- it
# compounds with an EXISTING mitigation in this same repo
# (QMiniLegPPORunnerCfg.clip_actions=3.0, see that field's comment), which
# was deliberately calibrated as "6 standard deviations of the
# init_noise_std=0.5 policy -- essentially never triggers under normal
# exploration". That calibration silently assumes std stays near 0.5. Once
# std has actually grown to, say, 4.4, a clip at 3.0 is barely half a
# standard deviation away and fires on most samples -- turning a rare,
# inert numerical backstop into a constantly-active distortion: the
# executed (post-clip) action stops matching the Gaussian the algorithm's
# log-probability/surrogate-loss math assumes was sampled, which is a
# known source of PPO instability in its own right. So the unbounded std
# and the fixed clip_actions bound were very plausibly reinforcing each
# other into the exact runaway spiral observed, not acting independently.
#
# Ceiling chosen as 1.0: comfortably above every noise_std level this
# project has ever seen during genuine, productive exploration (the best
# sustained-stepping-looking run peaked around 0.37-0.40; nothing legitimate
# has ever needed more), while keeping clip_actions=3.0 at least 3 standard
# deviations away even in the worst case -- still an uncommon tail event,
# not a routine one, preserving the original backstop's intent instead of
# just moving the same problem to a different number.
#
# Wired in via rsl_rl_qmini_leg_ppo_cfg.py setting
# RslRlPpoActorCriticCfg(class_name="ClampedActorCritic", ...) and
# monkeypatching this class into rsl_rl.runners.on_policy_runner's module
# namespace there -- see that file's comment for why the injection has to
# happen that specific way (on_policy_runner.py resolves class_name via a
# plain eval() against its own globals, so the class has to be visible
# there, not just importable from here).

import torch
from rsl_rl.modules import ActorCritic


class ClampedActorCritic(ActorCritic):
    max_action_std: float = 1.0

    def update_distribution(self, obs):
        # Clamp the PARAMETER itself (not just a local copy used for this
        # one call) so the optimizer can't keep pushing it further past the
        # ceiling step after step once it's already saturated there -- that
        # "phantom growth" would otherwise sit latent and immediately
        # re-explode the moment this clamp were ever loosened or removed.
        if self.noise_std_type == "scalar":
            with torch.no_grad():
                self.std.clamp_(max=self.max_action_std)
        elif self.noise_std_type == "log":
            with torch.no_grad():
                self.log_std.clamp_(max=torch.log(torch.tensor(self.max_action_std)))
        super().update_distribution(obs)
