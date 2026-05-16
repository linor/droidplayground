
[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.0-silver.svg)](https://github.com/isaac-sim/IsaacLab/tree/release/2.3.0)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)

## Droid Playground
This repository acts as a place to store out Reinforcement Learning experiments. The goal is to slowly increase each experiment in difficulty and gain new insights as we go along.

## Initial preparation
As suggested in the IsaacLab documentation, we use [Anaconda](https://www.anaconda.com/download). Start by creating an environment by running these commands in the root directory of the checkout:

```
conda env create --name droidplayground --file=environment.yml
conda activate droidplayground
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com 
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 
pip install dynamixel_sdk pre-commit
```

Note that these installation instructions work for Linux based operating systems, for others follow the [official installation instructions](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html), although the rest of the experiments are not guaranteed to work.

### Check IsaacSim installation
Before continueing, verify if the IsaacSim installation was successfull.

```
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
isaacsim
```

After accepting the EULA, the main IsaacSim window should show up with an empty stage. At this point your IsaacSim installation is working and you can close the window.

### IsaacLab installation 
Execute the followig commands to clone & install IsaacLab and prepare the Droid Playground environment.

```
git clone -b release/2.3.0 --single-branch https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
cd ..
python -m pip install -e rl/source/droidplayground
```

At this point, our reinforcement learning experiements should work. You can try this by listing the known environments:

```
python rl/scripts/list_envs.py
```

With this working, it's time to continue with [the servo experiment](experiments/servo/README.md)

