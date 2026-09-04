from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class QMiniLegPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 40000
    save_interval = 100
    experiment_name = "qmini-leg"

    # Added after a 40000-iteration run diverged catastrophically at the
    # very end: value_function_loss hit 1.7 BILLION, mean_reward crashed to
    # -28385, target_limit_penalty (quadratic and deliberately UNBOUNDED,
    # see that field's comment in qmini_leg_env.py) averaged 818 per step.
    # Root cause: clip_actions defaulted to None (RslRlOnPolicyRunnerCfg's
    # own default -- this class never overrode it), so nothing bounded a
    # raw policy output before it got scaled by action_scale into a joint
    # target. Once ANY env sampled one sufficiently large action (std=0.5
    # Gaussian noise makes this rare but not impossible over 40000
    # iterations x thousands of env-steps each), the resulting overshoot
    # fed into target_limit_penalty's unbounded square produced a reward
    # magnitude that dwarfed everything else, which blew up the critic's
    # value estimates -- which corrupts the advantage estimates for every
    # subsequent update, cascading into worse actions and a bigger
    # explosion. 3.0 is a generous bound (6 standard deviations of the
    # init_noise_std=0.5 policy -- essentially never triggers under normal
    # exploration, since the whole step-in-place gait only needs raw
    # actions in roughly [-0.4, 0.4] to reach its extremes) meant purely as
    # a numerical-stability backstop against outliers, not a behavioral
    # constraint. Paired with target_limit_penalty_max_overshoot_deg in
    # qmini_leg_env.py, which bounds the OTHER half of this same mechanism
    # (the penalty itself, regardless of what produced the oversized
    # target).
    clip_actions = 3.0

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[128, 128, 64],
        critic_hidden_dims=[128, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Raised from 0.005 -- the qmini-leg balance task saw a policy
        # collapse around iteration ~19k of a 30k run: action noise std
        # dropped from init 0.5 to ~0.01 and episode length collapsed to 1
        # step, with fall_rate=1.0 and near-zero value loss/surrogate loss
        # (i.e. the critic converged on "this always ends the same way").
        # Standing up on two legs from a free-floating spawn is hard enough
        # that early random exploration rarely if ever succeeds, so the
        # entropy bonus is what keeps the actor exploring past a run of bad
        # rollouts instead of shrinking its noise to minimize loss on a
        # reward landscape it can't yet improve on. 0.005 wasn't enough to
        # resist that -- 0.02 worked for standing.
        #
        # Raised again to 0.03 when moving from standing-still to the
        # step-in-place gait (keyframes_step_in_place_all_joints_2x_base_offset.json):
        # same collapse shape recurring one level up -- the policy settled
        # into an exaggerated weight-shift wobble that never actually lifts
        # a foot (visible as inflated roll tracking error, legs staying
        # grounded), because a genuine single-support step risks the same
        # fall/termination_penalty cost standing once did, and it's a new,
        # harder skill the standing-trained policy hasn't discovered yet.
        # Paired with temporarily disabling push disturbances
        # (push_interval_s in qmini_leg_env.py) and a margin on
        # orientation_reward (orientation_margin_deg) so a real step's
        # necessary lean isn't itself fought by the reward.
        #
        # LOWERED to 0.01 for resuming from a checkpoint that's already
        # found genuine stepping (once the foot_swing_target units bug was
        # fixed -- see keyframes_step_in_place_all_joints_2x_base_offset.json's
        # swing-target derivation in qmini_leg_env.py's __init__). 0.03
        # correctly escaped the early no-lift local optimum by ~iteration
        # 15-20k of that run (motion/left_swing_target and
        # motion/left_foot_clearance_cm became genuinely correlated), but
        # holding it that high for the REST of a 40k run then actively
        # eroded the policy it had already found: past ~iteration 27-30k,
        # Mean action noise std climbed to a new high (0.81, above its
        # entire earlier oscillation band), mean_reward went negative,
        # every per-joint tracking error got worse, and fall_rate across
        # saved checkpoints stopped improving monotonically and started
        # swinging wildly instead (9.65% -> 1.95% -> 0.78% -> 0.39% ->
        # 11.24% -> 0.00% -> 0.00% -> 2.73% -> 0.39%, measured via
        # compare_policies_isaaclab.py) -- exploration pressure that's
        # necessary to ESCAPE a bad local optimum becomes counterproductive
        # once a good one is already found and just needs consolidating.
        # This rsl_rl version has no entropy annealing/schedule (checked
        # rsl_rl/algorithms/ppo.py -- entropy_coef is a fixed scalar in the
        # loss the whole run), so dropping it manually on resume is the
        # only lever. Not all the way back to the original 0.005 that
        # caused the FIRST collapse (episode length 1, fall_rate 1.0) --
        # that was from a cold/random init where escaping bad optima
        # mattered most; this resume starts from an already-good policy,
        # so the failure mode this run needs guarding against is the
        # opposite one. Watch Mean action noise std again: it should now
        # settle rather than climb.
        #
        # RAISED BACK to 0.02 (from 0.01) -- 0.01 was ALSO too low: within
        # ~5700 iterations of resuming, Mean action noise std collapsed to
        # 0.03 and mean_episode_length pinned at 499/500 (max), i.e. a
        # THIRD distinct collapse shape from a THIRD different entropy_coef
        # value (0.005 -> immediate-fall collapse; 0.03 sustained too long
        # -> eroded an already-good policy; 0.01 -> collapsed to a frozen,
        # statically-tilted pose). This one was visually confirmed frozen
        # (two video frames 5 seconds apart were pixel-identical) and had a
        # specific signature: pitch/knee/ankle/yaw tracking were all
        # EXCELLENT (1-4 deg, best ever), but roll alone blew up (actual
        # right_roll=-5.77deg against a reference of +2.49deg -- a full
        # sign flip, not just extra amplitude) while foot_swing_reward
        # dropped from ~0.16 back to ~0.06 with negative foot clearance
        # despite a high swing_target -- i.e. the policy found it could
        # satisfy tracking_reward on 4 of 5 joint types while using an
        # un-referenced roll lean to avoid ever completing real weight
        # transfer. 0.02 (the value already proven to work for
        # consolidating the standing task) is the middle ground between a
        # value that just collapsed (0.01) and one that worked initially
        # then eroded a good policy over a full run (0.03). Paired this
        # time with re-enabling push disturbances (push_interval_s in
        # qmini_leg_env.py, back to ~4.0) -- disabling pushes protected
        # early exploration while stepping was still undiscovered, but now
        # that it demonstrably works, keeping them off removes the one
        # thing that makes "freeze in a static pose" actually risky. A
        # policy standing dead-still and never having to react to anything
        # has no pressure against exactly the collapse seen here.
        entropy_coef=0.02,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )