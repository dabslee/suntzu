"""Train a single MaskablePPO agent (guard vs. wolf by default) against the
rule-based enemy AI or a saved enemy model, then evaluate it for 20 episodes.

Uses action masking + several vectorized envs (batched policy inference) —
see README "Training performance" for why this is much faster than the old
single-env DQN setup.
"""
import os
import argparse
from typing import Optional

import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import FlattenObservation

from dnd_gym_env import DnDCombatEnv
from bestiary.guard import get_guard_stats
from bestiary.wolf import get_wolf_stats

# --- Configuration ---
# Guard vs. wolf is a winnable matchup, so the reward signal contains wins.
AGENT_STATS = get_guard_stats()
ENEMY_STATS = get_wolf_stats()
MAP_WIDTH = 10
MAP_HEIGHT = 10
N_ENVS = 8

LOG_DIR = "./dnd_ppo_tensorboard/"
MODEL_SAVE_PATH = "dnd_ppo_agent.zip"

os.makedirs(LOG_DIR, exist_ok=True)


def make_env(enemy_model_path_arg: Optional[str] = None,
             render_mode_arg: Optional[str] = None,
             verbose_arg: bool = True) -> DnDCombatEnv:
    return DnDCombatEnv(
        map_width=MAP_WIDTH,
        map_height=MAP_HEIGHT,
        agent_stats=AGENT_STATS,
        enemy_stats=ENEMY_STATS,
        render_mode=render_mode_arg,
        enemy_model_path=enemy_model_path_arg,
        verbose=verbose_arg
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MaskablePPO Agent for D&D Combat Environment.")
    parser.add_argument("--enemy_model_path", type=str, default=None,
                        help="Path to a pre-trained model for the enemy. If None, rule-based AI is used.")
    parser.add_argument("--total_timesteps", type=int, default=200000,
                        help="Total training timesteps. Default: 200000")
    parser.add_argument("--n_envs", type=int, default=N_ENVS,
                        help=f"Number of vectorized envs. Default: {N_ENVS}")
    args = parser.parse_args()

    if args.enemy_model_path and not os.path.exists(args.enemy_model_path):
        print(f"Warning: Enemy model file {args.enemy_model_path} not found! Using rule-based AI.")
        args.enemy_model_path = None
    print(f"Enemy: {'model ' + args.enemy_model_path if args.enemy_model_path else 'rule-based AI'}")

    print(f"Creating {args.n_envs} vectorized training environments...")
    # VecNormalize(norm_reward=True) rescales the +/-100 terminal rewards for
    # PPO's value function — without it PPO plateaus at ~0% wins. Observations
    # are NOT normalized, so the saved policy works standalone at inference.
    vec_env = VecNormalize(
        DummyVecEnv(
            # Only the first env copy logs its setup (they are identical)
            [(lambda v=(i == 0): FlattenObservation(
                make_env(enemy_model_path_arg=args.enemy_model_path, verbose_arg=v)))
             for i in range(args.n_envs)]),
        norm_obs=False, norm_reward=True, gamma=0.99)

    print("Defining the MaskablePPO model...")
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=256,        # rollout length per env (n_envs * n_steps per update)
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,      # entropy bonus for exploration
        tensorboard_log=LOG_DIR
    )

    print(f"Starting training for {args.total_timesteps} timesteps...")
    model.learn(total_timesteps=args.total_timesteps, log_interval=10, progress_bar=True)
    print("Training complete.")

    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    vec_env.close()

    # --- Evaluation ---
    print("Setting up evaluation environment...")
    eval_env_raw = make_env(enemy_model_path_arg=args.enemy_model_path)
    eval_env = FlattenObservation(eval_env_raw)

    print(f"Loading model for evaluation from {MODEL_SAVE_PATH}...")
    loaded_model = MaskablePPO.load(MODEL_SAVE_PATH)

    num_eval_episodes = 20
    total_rewards = 0.0
    wins = losses = draws = 0

    print(f"\nEvaluating for {num_eval_episodes} episodes "
          f"(max {eval_env_raw.max_episode_steps} steps each)...")
    for episode in range(num_eval_episodes):
        obs, info = eval_env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = loaded_model.predict(obs, deterministic=True,
                                             action_masks=eval_env_raw.action_masks())
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
        total_rewards += episode_reward

        if truncated and not terminated:
            draws += 1
            result = "Draw (max steps)"
        elif eval_env_raw.agent.is_alive and not eval_env_raw.enemy.is_alive:
            wins += 1
            result = "Agent Won"
        elif not eval_env_raw.agent.is_alive:
            losses += 1
            result = "Agent Lost"
        else:
            draws += 1
            result = "Draw"
        print(f"Episode {episode + 1}: {result}. Reward: {episode_reward:.2f}. "
              f"Agent HP: {eval_env_raw.agent.current_hp}, Enemy HP: {eval_env_raw.enemy.current_hp}")

    print(f"\n--- Evaluation Summary ---")
    print(f"Episodes: {num_eval_episodes}, Avg Reward: {total_rewards / num_eval_episodes:.2f}")
    print(f"Wins: {wins} ({wins / num_eval_episodes:.0%}), "
          f"Losses: {losses} ({losses / num_eval_episodes:.0%}), "
          f"Draws: {draws} ({draws / num_eval_episodes:.0%})")
    print(f"\nTensorBoard: tensorboard --logdir=\"{os.path.abspath(LOG_DIR)}\"")

    eval_env.close()
