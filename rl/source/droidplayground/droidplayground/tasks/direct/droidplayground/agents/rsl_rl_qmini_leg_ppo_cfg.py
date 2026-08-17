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
        # resist that -- try 0.02 and watch "Mean action noise std" over
        # training; if it's still trending toward ~0 well before the run
        # ends, this may need to go higher still (or combined with a
        # smaller termination_penalty_weight in qmini_leg_env.py -- see
        # that cfg field's comment).
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