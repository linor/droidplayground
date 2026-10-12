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
