import os
import argparse # For command-line arguments
from typing import Optional # For type hinting
import gymnasium as gym 
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
# evaluate_policy was removed in a previous step, manual loop is used
# from stable_baselines3.common.evaluation import evaluate_policy 
from gymnasium.wrappers import FlattenObservation 

# Assuming dnd_gym_env.py and bestiary are in the same directory or accessible in PYTHONPATH
from dnd_gym_env import DnDCombatEnv
from bestiary.commoner import get_commoner_stats
from bestiary.wolf import get_wolf_stats

# --- Configuration ---
AGENT_STATS = get_commoner_stats()
ENEMY_STATS = get_wolf_stats()
MAP_WIDTH = 10
MAP_HEIGHT = 10

LOG_DIR = "./dnd_dqn_tensorboard/"

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# --- Environment Setup ---
def make_env(enemy_model_path_arg: Optional[str] = None, render_mode_arg: Optional[str] = None):
    """Helper function to create an environment instance."""
    # print(f"make_env: enemy_model='{enemy_model_path_arg}', render='{render_mode_arg}'") # For debugging
    env = DnDCombatEnv(
        map_width=MAP_WIDTH,
        map_height=MAP_HEIGHT,
        agent_stats=AGENT_STATS,    # Uses global AGENT_STATS
        enemy_stats=ENEMY_STATS,    # Uses global ENEMY_STATS
        render_mode=render_mode_arg,
        enemy_model_path=enemy_model_path_arg # Pass the argument here
    )
    return env

# Create a function to properly wrap the environment
def wrap_env(env_to_wrap: gym.Env) -> gym.Env: 
    """Flattens the dictionary observation space."""
    wrapped_env = FlattenObservation(env_to_wrap)
    return wrapped_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN Agent for D&D Combat Environment.")
    parser.add_argument("--enemy_model_path", type=str, default=None,
                        help="Path to a pre-trained DQN model for the enemy. If None, rule-based AI is used.")
    parser.add_argument("--model_save_path", type=str, default=None,
                        help="Path to save resulting model.")
    args = parser.parse_args()
    MODEL_SAVE_PATH = args.model_save_path

    if args.enemy_model_path:
        print(f"Attempting to train agent against an enemy controlled by model: {args.enemy_model_path}")
        if not os.path.exists(args.enemy_model_path):
             print(f"Warning: Enemy model file {args.enemy_model_path} not found! Enemy will use rule-based AI.")
             args.enemy_model_path = None # Fallback to rule-based if model not found
    else:
        print("Training agent against a rule-based enemy AI.")

    print("Creating and wrapping the vectorized environment for training...")
    # Create a vectorized environment for SB3, passing the enemy_model_path
    vec_env = DummyVecEnv([lambda: FlattenObservation(make_env(enemy_model_path_arg=args.enemy_model_path, render_mode_arg=None))])
    print("Training environment created.")

    # --- Model Definition ---
    print("Defining the DQN model...")
    model = DQN(
        "MlpPolicy",
        vec_env,
        verbose=1, # Set to 1 to see training logs, 0 for less output
        learning_rate=1e-4,    # 0.0001
        buffer_size=50000,     # Size of the replay buffer
        learning_starts=1000,  # Number of steps before learning starts
        batch_size=32,         # Size of a batch sampled from replay buffer for training
        tau=1.0,               # Soft update coefficient (1 for hard update)
        gamma=0.99,            # Discount factor
        train_freq=4,          # Update the model every 4 steps (can be int or tuple)
        gradient_steps=1,      # How many gradient steps to perform when train_freq is met
        exploration_fraction=0.2, # Fraction of entire training period over which exploration rate is reduced
        exploration_final_eps=0.05, # Final value of random action probability
        tensorboard_log=LOG_DIR
    )
    print("DQN model defined.")

    # --- Training ---
    # TOTAL_TIMESTEPS = 50000 # Reduced for quicker initial testing (plan was 200k)
    TOTAL_TIMESTEPS = 1000 # Even shorter for a very quick smoke test
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        log_interval=10, # Log training info every 10 rollout collections (episodes for DQN)
        progress_bar=True # Show a progress bar during training
    )
    print("Training complete.")

    # --- Save Model ---
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("Model saved.")

    # --- Evaluation ---
    print("Setting up evaluation environment...")
    # For manual evaluation loop:
    # Pass the same enemy_model_path_arg for consistency in evaluation
    manual_eval_env_raw = make_env(enemy_model_path_arg=args.enemy_model_path, render_mode_arg=None) 
    manual_eval_env_wrapped = wrap_env(manual_eval_env_raw) 
    
    # Load the model that was just trained and saved
    print(f"Loading model for manual evaluation from {MODEL_SAVE_PATH}...")
    loaded_model = DQN.load(MODEL_SAVE_PATH)

    num_eval_episodes = 20 
    total_rewards_manual = 0
    wins = 0
    losses = 0
    draws = 0
    
    print(f"\nStarting manual evaluation for {num_eval_episodes} episodes (max steps per episode: {manual_eval_env_raw.max_episode_steps})...")

    for episode in range(num_eval_episodes):
        print(f"--- Starting evaluation episode {episode + 1}/{num_eval_episodes} ---")
        obs, info = manual_eval_env_wrapped.reset()
        
        terminated = False 
        truncated = False
        done = False
        episode_reward = 0
        episode_steps = 0 

        while not done:
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = manual_eval_env_wrapped.step(action) 
            
            episode_steps = manual_eval_env_raw.current_episode_steps 
            
            done = terminated or truncated
            episode_reward += reward
        
        total_rewards_manual += episode_reward
        print(f"Episode {episode + 1} finished after {episode_steps} steps. Reward: {episode_reward:.2f}")

        if truncated and not terminated: 
            draws += 1
            print(f"Result: Draw (reached max {manual_eval_env_raw.max_episode_steps} steps). Agent HP: {manual_eval_env_raw.agent.current_hp}, Enemy HP: {manual_eval_env_raw.enemy.current_hp}")
        elif terminated: 
            if manual_eval_env_raw.agent.is_alive and not manual_eval_env_raw.enemy.is_alive:
                wins += 1
                print(f"Result: Agent Won. Agent HP: {manual_eval_env_raw.agent.current_hp}, Enemy HP: {manual_eval_env_raw.enemy.current_hp}")
            elif not manual_eval_env_raw.agent.is_alive: 
                losses += 1
                print(f"Result: Agent Lost. Agent HP: {manual_eval_env_raw.agent.current_hp}, Enemy HP: {manual_eval_env_raw.enemy.current_hp}")
            else: 
                draws +=1 
                print(f"Result: Undetermined (Terminated but not clear win/loss by HP - counted as draw). Agent HP: {manual_eval_env_raw.agent.current_hp}, Enemy HP: {manual_eval_env_raw.enemy.current_hp}")

    avg_reward_manual = total_rewards_manual / num_eval_episodes if num_eval_episodes > 0 else 0
    win_rate = wins / num_eval_episodes if num_eval_episodes > 0 else 0
    loss_rate = losses / num_eval_episodes if num_eval_episodes > 0 else 0
    draw_rate = draws / num_eval_episodes if num_eval_episodes > 0 else 0

    print(f"\n--- Manual Evaluation Summary ---")
    print(f"Total Episodes: {num_eval_episodes}")
    print(f"Average Reward: {avg_reward_manual:.2f}")
    print(f"Wins: {wins} ({win_rate:.2%})")
    print(f"Losses: {losses} ({loss_rate:.2%})")
    print(f"Draws (includes max steps and other terminated draws): {draws} ({draw_rate:.2%})")
    print("------------------------------------")

    print("\nTo view tensorboard logs, run the following command in your terminal:")
    print(f"tensorboard --logdir=\"{os.path.abspath(LOG_DIR)}\"")
    print("\nTraining and evaluation script example complete.")

    # --- Cleanup ---
    vec_env.close() # This is the training env
    manual_eval_env_wrapped.close() # Close the evaluation environment
