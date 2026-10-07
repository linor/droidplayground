# Subclasses rsl_rl's PPO purely to put a hard ceiling on the adaptive
# learning rate, which the library caps at 1e-2 (hard-coded in
# rsl_rl/algorithms/ppo.py's update(): lr *= 1.5 whenever the minibatch KL
# is below desired_kl / 2, lr /= 1.5 when above 2 * desired_kl).
#
# Added 2026-10-07 after run 2026-10-06_12-09-36 collapsed in ONE update at
# iteration 228519, ~28k iterations into an otherwise healthy resume: the
# logged learning rate went 1e-4 -> 1.3e-3 -> 2.9e-3 -> 6.6e-3 over a few
# iterations, the next update's surrogate loss jumped to 0.22 and noise std
# 0.074 -> 0.081, and the iteration after that every env fell (episode
# length 488 -> 35, value loss ~300 -> 100,000). The run then relearned a
# worse gait with the left foot no longer lifting. The adaptation runs on
# EVERY minibatch (num_learning_epochs * num_mini_batches = 20 per
# iteration), so with a converged, low-noise policy (tiny KL) the rate can
# grow 1.5^20 ~ 3000x within a single iteration until it hits 1e-2 -- one
# update at that rate is enough to destroy the policy.
#
# Ceiling: MAX_LEARNING_RATE = 1e-3, the configured starting rate and above
# anything the healthy part of that run needed (it hovered ~1e-4..1.3e-3),
# so normal adaptation (including going DOWN) is unchanged. Implemented as
# a property so PPO's own `self.learning_rate = ...` assignments (in
# __init__ and in update()'s schedule) are clamped and the clamped value is
# what update() then writes into the optimizer's param groups.
from rsl_rl.algorithms import PPO

MAX_LEARNING_RATE = 1.0e-3


class CappedLrPPO(PPO):
    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self._learning_rate = min(float(value), MAX_LEARNING_RATE)
