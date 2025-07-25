#!/usr/bin/env python3
"""
Working Ray RLLib CartPole using Ray Tune (based on working pendulum.py).
This approach uses hyperparameter tuning which works better with current Ray versions.
Enhanced to use hoppetusse cluster via Ray Client.
"""

import ray
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig

print("🚀 Ray RLLib CartPole with Tune")
print("=" * 32)

# Connect to hoppetusse cluster via Ray Client
try:
    ray.init("ray://localhost:10001")
    cluster_resources = ray.cluster_resources()
    total_cpus = int(cluster_resources.get('CPU', 0))
    total_memory = cluster_resources.get('memory', 0) / (1024**3)
    total_gpus = int(cluster_resources.get('GPU', 0))
    
    print(f"✅ Connected to hoppetusse cluster!")
    print(f"   CPUs: {total_cpus}, Memory: {total_memory:.1f}GB, GPUs: {total_gpus}")
except Exception as e:
    print("Exiting, cant connect to cluster")
    import sys
    sys.exit()


print("Target: 195+ average reward (CartPole solved)")
print("Testing 3 learning rates with hyperparameter sweep")
print(f"Max concurrent trials: {total_cpus}")
print("-" * 32)

config = (
    PPOConfig()
    .environment("CartPole-v1")
    # Simple hyperparameter sweep
    .training(
        lr=tune.grid_search([0.001, 0.0005, 0.0001]),
    )
)

# Create a Tuner instance to manage the trials
tuner = tune.Tuner(
    config.algo_class,
    param_space=config,
    # Stop when CartPole is solved (195+ average reward)
    run_config=train.RunConfig(
        stop={"env_runners/episode_return_mean": 195.0},
    ),
    tune_config=tune.TuneConfig(
        max_concurrent_trials=min(total_cpus, 3),  # Use cluster CPUs but not more than we have trials
    ),
)

# Run the Tuner and capture the results
results = tuner.fit()

print("\n✅ Tuning completed!")

# Get the best trial based on the stopping metric
best_result = results.get_best_result(metric="env_runners/episode_return_mean", mode="max", scope="last-5-avg")

print(f"\n🏆 Best Trial:")
print(f"   Learning Rate: {best_result.config['lr']}")

# Access the metric value directly from the Result object
try:
    final_reward = best_result.metrics["env_runners/episode_return_mean"]
    print(f"   Final Reward: {final_reward:.1f}")
except KeyError:
    # Fallback: show all available metrics
    print(f"   Available metrics: {list(best_result.metrics.keys())}")
    
print(f"   Trial Path: {best_result.path}")
