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
    print("Starting evaluation with manual loop...")
    # Create a new environment for evaluation - non-vectorized, but wrapped
    manual_eval_env_raw = make_env()
    eval_env_wrapped = wrap_env(manual_eval_env_raw) # wrap_env uses FlattenObservation

    # Load the model
    loaded_model = DQN.load(MODEL_SAVE_PATH)

    num_eval_episodes = 20 
    total_rewards_manual = 0 # Renamed for clarity, matching original commented out code
    wins = 0
    losses = 0
    draws = 0
    
    print(f"\nStarting manual evaluation for {num_eval_episodes} episodes (max steps per episode: {manual_eval_env_raw.max_episode_steps})...")

    for episode in range(num_eval_episodes):
        print(f"--- Starting evaluation episode {episode + 1}/{num_eval_episodes} ---")
        obs, info = eval_env_wrapped.reset()
        
        # These flags are from the last step of an episode
        # Initialize them before the loop for clarity, will be updated by step()
        terminated = False 
        truncated = False
        done = False
        episode_reward = 0
        episode_steps = 0 

        while not done:
            action, _states = loaded_model.predict(obs, deterministic=True)
            # Store the flags from the step
            obs, reward, terminated, truncated, info = eval_env_wrapped.step(action) 
            
            episode_steps = manual_eval_env_raw.current_episode_steps 
            
            done = terminated or truncated
            episode_reward += reward
        
        total_rewards_manual += episode_reward
        print(f"Episode {episode + 1} finished after {episode_steps} steps. Reward: {episode_reward:.2f}")

        # Check episode outcome using the 'terminated' and 'truncated' flags from the last step
        if truncated and not terminated: # Check for truncation first
            draws += 1
            print(f"Result: Draw (reached max {manual_eval_env_raw.max_episode_steps} steps). Agent HP: {manual_eval_env_raw.agent.current_hp}, Enemy HP: {manual_eval_env_raw.enemy.current_hp}")
        elif terminated: # If not a draw by truncation, check for termination by game rules
            if manual_eval_env_raw.agent.is_alive and not manual_eval_env_raw.enemy.is_alive:
                wins += 1
                print(f"Result: Agent Won. Agent HP: {manual_eval_env_raw.agent.current_hp}, Enemy HP: {manual_eval_env_raw.enemy.current_hp}")
            elif not manual_eval_env_raw.agent.is_alive: 
                losses += 1
                print(f"Result: Agent Lost. Agent HP: {manual_eval_env_raw.agent.current_hp}, Enemy HP: {manual_eval_env_raw.enemy.current_hp}")
            else: # Terminated, but agent is alive AND enemy is alive (or both dead - less likely scenario)
                  # This case could be a special draw or an unexpected termination.
                draws +=1 # Counting as a draw for now if not a clear win/loss upon termination
                print(f"Result: Undetermined (Terminated but not clear win/loss by HP - counted as draw). Agent HP: {manual_eval_env_raw.agent.current_hp}, Enemy HP: {manual_eval_env_raw.enemy.current_hp}")
        # No 'else' needed here, as 'done' must be True for the loop to exit.

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
    eval_env_wrapped.close() # Close the evaluation environment
