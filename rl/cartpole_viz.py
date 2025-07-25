#!/usr/bin/env python3
"""
Simple CartPole visualization script using gymnasium.
This script runs CartPole with random actions for demonstration purposes.
"""

import gymnasium as gym
import numpy as np
import time


def run_cartpole_demo(episodes=5, max_steps=200, delay=0.05):
    """
    Run CartPole environment with random actions for visualization.
    
    Args:
        episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        delay: Delay between steps for visualization
    """
    # Create CartPole environment with human rendering
    env = gym.make('CartPole-v1', render_mode='human')
    
    print(f"CartPole Environment Info:")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action meanings: 0=Left, 1=Right")
    print(f"\nObservation meanings:")
    print(f"  [0] Cart Position")
    print(f"  [1] Cart Velocity") 
    print(f"  [2] Pole Angle")
    print(f"  [3] Pole Angular Velocity")
    print()
    
    total_rewards = []
    
    for episode in range(episodes):
        observation, info = env.reset()
        episode_reward = 0
        
        print(f"Episode {episode + 1}:")
        print(f"  Initial observation: {observation}")
        
        for step in range(max_steps):
            # Take random action
            action = env.action_space.sample()
            
            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Print step info
            if step % 20 == 0:  # Print every 20 steps to avoid spam
                print(f"  Step {step:3d}: action={action}, reward={reward:.1f}, obs={observation}")
            
            # Add delay for visualization
            time.sleep(delay)
            
            # Check if episode is done
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        print(f"  Episode ended after {step + 1} steps with reward: {episode_reward}")
        print(f"  Final observation: {observation}")
        print()
    
    env.close()
    
    # Print summary statistics
    print(f"Summary after {episodes} episodes:")
    print(f"  Average reward: {np.mean(total_rewards):.1f}")
    print(f"  Best reward: {np.max(total_rewards):.1f}")
    print(f"  Worst reward: {np.min(total_rewards):.1f}")
    print(f"  Std deviation: {np.std(total_rewards):.1f}")


def analyze_observation_space():
    """
    Analyze the CartPole observation space bounds and characteristics.
    """
    env = gym.make('CartPole-v1')
    
    print("CartPole Observation Space Analysis:")
    print("=" * 50)
    
    obs_space = env.observation_space
    print(f"Shape: {obs_space.shape}")
    print(f"Low bounds: {obs_space.low}")
    print(f"High bounds: {obs_space.high}")
    print(f"Data type: {obs_space.dtype}")
    
    print("\nObservation Details:")
    labels = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
    units = ["m", "m/s", "rad", "rad/s"]
    
    for i, (label, unit, low, high) in enumerate(zip(labels, units, obs_space.low, obs_space.high)):
        if np.isinf(low) or np.isinf(high):
            bound_str = "unbounded"
        else:
            bound_str = f"[{low:.2f}, {high:.2f}]"
        print(f"  [{i}] {label:<20} ({unit:<5}) Range: {bound_str}")
    
    env.close()


if __name__ == "__main__":
    print("CartPole Visualization Demo")
    print("=" * 40)
    
    # First analyze the environment
    analyze_observation_space()
    print()
    
    # Ask user for demo parameters
    try:
        episodes = int(input("Number of episodes to run (default 3): ") or "3")
        delay = float(input("Delay between steps in seconds (default 0.05): ") or "0.05")
    except (ValueError, KeyboardInterrupt):
        episodes = 3
        delay = 0.05
    
    print(f"\nRunning {episodes} episodes with {delay}s delay...")
    print("Close the pygame window to stop early.\n")
    
    try:
        run_cartpole_demo(episodes=episodes, delay=delay)
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError during demo: {e}")
        print("Make sure you have pygame installed and display is available.")