import sys
import os
import argparse
from stable_baselines3 import DQN
from gymnasium.wrappers import FlattenObservation
from gymnasium.utils.env_checker import check_env

from dnd_gym_env import DnDCombatEnv 
from bestiary.commoner import get_commoner_stats
from bestiary.wolf import get_wolf_stats

MODEL_PATH = "dnd_dqn_agent.zip" # Path to the saved model

def main(AGENT_TYPE):
    """
    Runs the D&D Combat Environment with specified agent type.
    """
    print(f"Initializing D&D Combat Environment with {AGENT_TYPE} agent...")

    agent_stats = get_commoner_stats()
    enemy_stats = get_wolf_stats()

    # Create the raw environment - this instance will be used throughout.
    env = DnDCombatEnv(
        map_width=10, map_height=10,
        agent_stats=agent_stats, enemy_stats=enemy_stats,
        render_mode='human' 
    )
    print("Environment instantiated.")

    # Run Environment Checker on the raw environment
    try:
        print("\nRunning Environment Checker...")
        check_env(env) 
        print("Environment check passed successfully!")
    except Exception as e:
        print(f"Environment check failed: {e}")
        env.close()
        return
    print("Render mode: ", env.render_mode)

    model = None
    flatten_wrapper = None # Will wrap 'env' if using a trained agent

    if AGENT_TYPE == "trained":
        if not os.path.exists(MODEL_PATH):
            print(f"Warning: Model file not found at {MODEL_PATH}. Falling back to random agent.")
            AGENT_TYPE = "random" # Fallback
        else:
            print(f"Loading trained model from {MODEL_PATH}...")
            try:
                model = DQN.load(MODEL_PATH)
                print("Model loaded.")
                # The flatten_wrapper should wrap the *same instance* of env that is used in the loop
                flatten_wrapper = FlattenObservation(env)
            except Exception as e:
                print(f"Error loading model: {e}. Falling back to random agent.")
                AGENT_TYPE = "random"
                model = None # Ensure model is None if loading failed

    print(f"\nStarting interaction loop with {AGENT_TYPE} agent...")
    try:
        # Reset the raw environment. The observation is a dict.
        obs_dict, info = env.reset(seed=42) 
    except Exception as e:
        print(f"Error during env.reset(): {e}")
        env.close()
        return
        
    env.render() # Initial render

    max_run_steps = 150 # Increased steps for potentially longer trained agent episodes
    for step_num in range(max_run_steps):
        print(f"\n--- Step {step_num + 1} ---")

        action = None
        action_desc_str = "N/A"

        if AGENT_TYPE == "trained" and model and flatten_wrapper:
            try:
                # Flatten the dictionary observation for the model
                flat_obs_for_model = flatten_wrapper.observation(obs_dict)
                action, _ = model.predict(flat_obs_for_model, deterministic=True)
                action_desc_str = f"Trained Agent Action (Index {action}): {env._decode_action(action).get('name', 'Unknown')}"
            except Exception as e:
                print(f"Error during model prediction: {e}. Using random action instead.")
                AGENT_TYPE = "random" # Fallback for this step
        
        if AGENT_TYPE == "random": # Also handles fallback
            action = env.action_space.sample()
            action_desc_str = f"Random Agent Action (Index {action}): {env._decode_action(action).get('name', 'Unknown')}"
        
        print(action_desc_str)

        if action is None: # Should not happen if logic is correct
            print("Error: No action determined. Breaking loop.")
            break

        try:
            obs_dict, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            print(f"Error during env.step() with action {action}: {e}")
            import traceback
            traceback.print_exc()
            break
            
        env.render()
        
        print(f"Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
        print(f"Agent HP: {env.agent.current_hp}, Enemy HP: {env.enemy.current_hp}")
        print(f"Agent Pos: {env.agent.position}, Enemy Pos: {env.enemy.position}")
        print(f"Steps taken in episode: {env.current_episode_steps}")


        if terminated or truncated:
            print("\nEpisode finished!")
            if terminated:
                if env.agent.is_alive and not env.enemy.is_alive: print("Result: Agent Won!")
                elif not env.agent.is_alive: print("Result: Agent Lost!")
                else: print("Result: Terminated for other reasons.")
            if truncated:
                print(f"Result: Draw (reached max {env.max_episode_steps} steps).")
            
            # Option to run more episodes or break
            # For now, break after one episode in this script's run
            break 
    
    if step_num == max_run_steps - 1 and not (terminated or truncated):
        print(f"\nReached max run steps ({max_run_steps}) for this interaction without termination/truncation.")

    env.close()
    print("\nEnvironment closed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run D&D Combat Environment with a random or trained agent.")
    parser.add_argument("--agent", type=str, default="random", choices=["random", "trained"],
                        help="Type of agent to run (random or trained). Default: random")
    args = parser.parse_args()
    
    # Assuming dnd_gym_env.py, bestiary/*, and dice.py are structured to be found.
    # If run_env.py is in the root, and dnd_gym_env is in root, and bestiary is a dir in root:
    # no special sys.path manipulation should be needed if running from root.
    main(AGENT_TYPE="trained")
