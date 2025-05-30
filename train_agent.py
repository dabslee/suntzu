import os
import gymnasium as gym # For type hinting and FlattenObservation
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import FlattenObservation # Correct import for Gymnasium

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
MODEL_SAVE_PATH = "dnd_dqn_agent.zip"

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# --- Environment Setup ---
def make_env():
    """Helper function to create an environment instance."""
    env = DnDCombatEnv(
        map_width=MAP_WIDTH,
        map_height=MAP_HEIGHT,
        agent_stats=AGENT_STATS,
        enemy_stats=ENEMY_STATS,
        render_mode=None # No rendering during training for speed
    )
    return env

# Create a function to properly wrap the environment (optional, can be done in-line)
def wrap_env(env_to_wrap: gym.Env) -> gym.Env: # Added type hint for clarity
    """Flattens the dictionary observation space."""
    wrapped_env = FlattenObservation(env_to_wrap)
    return wrapped_env

if __name__ == "__main__":
    print("Creating and wrapping the vectorized environment...")
    # Create a vectorized environment for SB3
    # This combines make_env and FlattenObservation for each environment instance in the vector
    vec_env = DummyVecEnv([lambda: FlattenObservation(make_env())])
    print("Environment created and wrapped.")

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
    TOTAL_TIMESTEPS = 50000 # Reduced for quicker initial testing (plan was 200k)
    # TOTAL_TIMESTEPS = 1000 # Even shorter for a very quick smoke test
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
    print("Starting evaluation...")
    # Create a new vectorized environment for evaluation for consistency
    eval_vec_env = DummyVecEnv([lambda: FlattenObservation(make_env())])

    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_vec_env, 
        n_eval_episodes=100, 
        deterministic=True, 
        warn=False # Suppress warnings during evaluation
    )
    print(f"Evaluation using SB3 evaluate_policy: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Manual evaluation loop for custom metrics like win rate
    num_eval_episodes = 100
    total_rewards_manual = 0
    wins = 0
    draws = 0
    print("DEBUG: About to load model...")
    loaded_model = DQN.load(MODEL_SAVE_PATH)
    print("DEBUG: Model loaded successfully.")
    
    print("DEBUG: About to create raw manual eval env...")
    manual_eval_env_raw = make_env()
    print("DEBUG: Raw manual eval env created.")

    print("DEBUG: About to wrap manual eval env...")
    manual_eval_env_wrapped = wrap_env(manual_eval_env_raw) # wrap_env is FlattenObservation
    print("DEBUG: Manual eval env wrapped.")

    print(f"Starting manual evaluation for {num_eval_episodes} episodes for win rate...") # You see this line

    for episode in range(num_eval_episodes):
        # It's good practice to have the episode start print here:
        # print(f"--- Starting evaluation episode {episode + 1}/{num_eval_episodes} ---") 
        # print(f"DEBUG: Episode {episode + 1}: About to reset env...")
        obs, info = manual_eval_env_wrapped.reset()
        # print(f"DEBUG: Episode {episode + 1}: Env reset successfully.")

        done = False
        episode_reward = 0
        while not done:
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = manual_eval_env_wrapped.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        # Access the raw environment to check win condition
        if manual_eval_env_raw.agent.is_alive and not manual_eval_env_raw.enemy.is_alive:
            wins += 1
        if manual_eval_env_raw.agent.is_alive and manual_eval_env_raw.enemy.is_alive:
            draws += 1
        total_rewards_manual += episode_reward
        print(f"Manual Eval - Episode {episode + 1}: Reward = {episode_reward:.2f}, Agent HP = {manual_eval_env_raw.agent.current_hp}, Enemy HP = {manual_eval_env_raw.enemy.current_hp}, Agent Alive: {manual_eval_env_raw.agent.is_alive}, Enemy Alive: {not manual_eval_env_raw.enemy.is_alive}")


    avg_reward_manual = total_rewards_manual / num_eval_episodes
    win_rate = wins / num_eval_episodes
    draw_rate = draws / num_eval_episodes
    print(f"Manual Evaluation Complete: Average Reward = {avg_reward_manual:.2f}, Win Rate = {win_rate:.2%}, Draw Rate = {draw_rate:.2%}")

    print("\nTo view tensorboard logs, run the following command in your terminal:")
    print(f"tensorboard --logdir=\"{os.path.abspath(LOG_DIR)}\"")
    print("\nTraining and evaluation script example complete.")

    # --- Cleanup ---
    vec_env.close()
    eval_vec_env.close()
    manual_eval_env_wrapped.close() # This will call close on manual_eval_env_raw too if properly chained.
                                 # Or call manual_eval_env_raw.close() explicitly if needed.
