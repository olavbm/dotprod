# Reinforcement Learning Experiments

This directory contains reinforcement learning experiments using gymnasium environments and the custom autograd system.

## Contents

- `cartpole_viz.py` - CartPole environment visualization with random actions
- `rllib_cartpole.py` - Ray RLLib PPO implementation with UV runtime hook (complex integration)
- `simple_rllib.py` - Simplified Ray RLLib script with local mode
- `ray_uv_cartpole.py` - Ray local mode experiment (working on UV integration)
- `ppo_simple.py` - Pure PyTorch PPO implementation ✅ **WORKING SOLUTION**
- Future: Custom policy gradient implementations using the autograd system

## Getting Started

Install RL dependencies:
```bash
uv sync --extra rl
```

For CPU-only PyTorch (much smaller download):
```bash
uv sync --extra rl --index-url https://download.pytorch.org/whl/cpu
```

Run CartPole visualization:
```bash
uv run rl/cartpole_viz.py
```

Train CartPole with PyTorch PPO (working solution):
```bash
uv run rl/ppo_simple.py
```

## Ray RLLib Integration Status

**Note**: Ray RLLib integration with UV is complex due to runtime environment issues. While the UV runtime hook approach from Anyscale works for distributed clusters, local development has challenges with working directory validation.

Try Ray RLLib (experimental):
```bash
uv run rl/rllib_cartpole.py --mode train --iterations 10
```

**Recommended approach**: Use the working PyTorch PPO implementation:
```bash
uv run rl/ppo_simple.py
```

## Environment Details

### CartPole-v1
- **Action space**: Discrete(2) - 0=Left, 1=Right
- **Observation space**: Box(4) - [cart_pos, cart_vel, pole_angle, pole_angular_vel]
- **Termination**: Pole angle > 15°, cart position > 2.4, or 500 steps
- **Reward**: +1 for each step the pole remains upright

## Planned Implementations

1. **REINFORCE** - Policy gradient method using custom autograd
2. **Actor-Critic** - Value function estimation with policy optimization
3. **Q-Learning** - Value-based method with neural network approximation