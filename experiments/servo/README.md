## Servo Experiment
Validating our Reinforcement Learning pipeline was our top priority, however, most experiments were too complex to easily validate where things went wrong. We needed the 'Hello World' of RL and ended up replicating a servo.

## Reinforcment Learning Task

We've created a task 'DroidPlayground-Servo' that captures the task of learning how to drive a velocity driven motor towards a target angle. 

### Relevant files
Check the following files and follow the relevant imports:

* source/droidplayground/droidplayground/assets/xl330.py
* source/droidplayground/droidplayground/tasks/manager_based/droidplayground/servo_env_cfg.py

### Commands
A single command generates a target angle at random intervals. For this no existing command existed, so a custom command was created.

### Observations
Angle Error
: Difference in radians between the target angle and the current angle of the "servo"

Joint Velocity
: Speed of the rotation (might not even be needed)

### Actions
A single action (JointVelocityActionCfg) was configured to actuate the joint

### Rewards
* Angle error
* Joint velocity (shaping reward)

### Train policy
Run the RSL_RL training script for 1000 cycles to train a decent policy:

```
python rl/scripts/rsl_rl/train.py --task DroidPlayground-Servo --num_envs 4096 --headless
```

After training is complete, you can replay the latest policy by executing the following command. This should show some simulated servos trying to track a random angle.

```
python rl/scripts/rsl_rl/play.py --task DroidPlayground-Servo --num_envs 9
```

This will also export the latest policy in the logs/rsl_rl/servo/__(timestamp)__/exported folder.


## Transfering to real device

The trained policy can then be ran using a simple python script, with for instance two Dynamixel XL330-M288-T servos. 

Copy the policy.pt file from the above mentioned exported folder to the experiments/servo folder.

```
python ./test_policy_outputs.py --policy policy.pt
```

That should print a result similar to the following:

| error_deg | error_rad | raw_action | clipped_action | velocity_rad_s |
| --------- | --------- | ---------- | -------------- | -------------- |
| -180.0    | -3.1416   | -0.7513    | -0.7513        | -8.1144        |
| -165.0    | -2.8798   | -0.7053    | -0.7053        | -7.6174        |
| -150.0    | -2.6180   | -0.6626    | -0.6626        | -7.1564        |
| -135.0    | -2.3562   | -0.6222    | -0.6222        | -6.7198        |
| -120.0    | -2.0944   | -0.5831    | -0.5831        | -6.2976        |
| -105.0    | -1.8326   | -0.5444    | -0.5444        | -5.8799        |
| -90.0     | -1.5708   | -0.5052    | -0.5052        | -5.4563        |
| -75.0     | -1.3090   | -0.4642    | -0.4642        | -5.0135        |
| -60.0     | -1.0472   | -0.4197    | -0.4197        | -4.5325        |
| -45.0     | -0.7854   | -0.3686    | -0.3686        | -3.9812        |
| -30.0     | -0.5236   | -0.2994    | -0.2994        | -3.2335        |
| -15.0     | -0.2618   | -0.1940    | -0.1940        | -2.0954        |
| +0.0      | +0.0000   | -0.0441    | -0.0441        | -0.4765        |
| +15.0     | +0.2618   | +0.1068    | +0.1068        | +1.1532        |
| +30.0     | +0.5236   | +0.2014    | +0.2014        | +2.1749        |
| +45.0     | +0.7854   | +0.2586    | +0.2586        | +2.7931        |
| +60.0     | +1.0472   | +0.2890    | +0.2890        | +3.1211        |
| +75.0     | +1.3090   | +0.2974    | +0.2974        | +3.2118        |
| +90.0     | +1.5708   | +0.2852    | +0.2852        | +3.0800        |
| +105.0    | +1.8326   | +0.2538    | +0.2538        | +2.7411        |
| +120.0    | +2.0944   | +0.2154    | +0.2154        | +2.3262        |
| +135.0    | +2.3562   | +0.1726    | +0.1726        | +1.8643        |
| +150.0    | +2.6180   | +0.1266    | +0.1266        | +1.3674        |
| +165.0    | +2.8798   | +0.0783    | +0.0783        | +0.8459        |
| +180.0    | +3.1416   | +0.0284    | +0.0284        | +0.3063        |

To try the policy on a real device, connect your Dynamixel U2D2 and run the 'mirror_policy_two_xl330.py' script.

## TODO
- Actuator damping/effort_limit needed to be increased to make the joint move at a reasonable speed. We should investigate what causes the normal values to not be sufficient (potential some USD configuration)
- Investigate the error 'PhysX error: PxRigidBody::setMassSpaceInertiaTensor(): components must be > 0 for articulations, FILE /builds/omniverse/physics/physx/source/physx/src/NpRigidBodyTemplate.h, LINE 460'
