# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="DroidPlayground-QMini-Leg",
    entry_point=f"{__name__}.qmini_leg_env:QminiLegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_leg_env:QminiLegEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_qmini_leg_ppo_cfg:QMiniLegPPORunnerCfg",
    },
)

# Omnidirectional walking from the stylized_gait reference library (same env
# class, QminiWalkEnvCfg sets gait_library_path; observation 31 = step-in-place
# 28 + the command vx, vy, wz).
gym.register(
    id="DroidPlayground-QMini-Walk",
    entry_point=f"{__name__}.qmini_leg_env:QminiLegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_leg_env:QminiWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_qmini_leg_ppo_cfg:QMiniWalkPPORunnerCfg",
    },
)

# Walking stage 2: resume from a stage-1 run with the four late penalties back
# on and the exploration std capped at 0.5 (see QminiWalkStage2EnvCfg).
gym.register(
    id="DroidPlayground-QMini-Walk-Stage2",
    entry_point=f"{__name__}.qmini_leg_env:QminiLegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_leg_env:QminiWalkStage2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_qmini_leg_ppo_cfg:QMiniWalkStage2PPORunnerCfg",
    },
)

# Walking stage 5: gait library v7 (lateral pelvis sway) at the 2x cadence
# (from stage 3) and at the 1.5x cadence (from stage 4).
gym.register(
    id="DroidPlayground-QMini-Walk-Stage5",
    entry_point=f"{__name__}.qmini_leg_env:QminiLegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_leg_env:QminiWalkStage5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_qmini_leg_ppo_cfg:QMiniWalkStage2PPORunnerCfg",
    },
)
gym.register(
    id="DroidPlayground-QMini-Walk-Stage5-Cadence15",
    entry_point=f"{__name__}.qmini_leg_env:QminiLegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_leg_env:QminiWalkStage5Cadence15EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_qmini_leg_ppo_cfg:QMiniWalkStage2PPORunnerCfg",
    },
)

# Walking stage 8: stage 7 with foot heights from the ground (no stance capture).
gym.register(
    id="DroidPlayground-QMini-Walk-Stage8",
    entry_point=f"{__name__}.qmini_leg_env:QminiLegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_leg_env:QminiWalkStage8EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_qmini_leg_ppo_cfg:QMiniWalkStage2PPORunnerCfg",
    },
)

# Walking stage 7: stage 6 + a left/right swing symmetry penalty.
gym.register(
    id="DroidPlayground-QMini-Walk-Stage7",
    entry_point=f"{__name__}.qmini_leg_env:QminiLegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_leg_env:QminiWalkStage7EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_qmini_leg_ppo_cfg:QMiniWalkStage2PPORunnerCfg",
    },
)

# Walking stage 6: stage 5 (1.5x, v7) + the deploy's rate limit, a roll-bias
# penalty and more left/right asymmetry in the randomization.
gym.register(
    id="DroidPlayground-QMini-Walk-Stage6",
    entry_point=f"{__name__}.qmini_leg_env:QminiLegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_leg_env:QminiWalkStage6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_qmini_leg_ppo_cfg:QMiniWalkStage2PPORunnerCfg",
    },
)

# Walking stage 4: stage 3 at the 1.5x cadence (less body roll), same commands.
gym.register(
    id="DroidPlayground-QMini-Walk-Stage4",
    entry_point=f"{__name__}.qmini_leg_env:QminiLegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_leg_env:QminiWalkStage4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_qmini_leg_ppo_cfg:QMiniWalkStage2PPORunnerCfg",
    },
)

# Walking stage 3: the full command range, resumed from stage 2 (same runner).
gym.register(
    id="DroidPlayground-QMini-Walk-Stage3",
    entry_point=f"{__name__}.qmini_leg_env:QminiLegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_leg_env:QminiWalkStage3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_qmini_leg_ppo_cfg:QMiniWalkStage2PPORunnerCfg",
    },
)
