import argparse
import importlib
import os
import time # For progress updates

# Assuming dnd_gym_env and SB3 are importable
# Add try-except for SB3 if it's optional, though required for this script
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_util import make_vec_env # If needed, but likely direct env use
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_util import make_vec_env # If needed, but likely direct env use
    from gymnasium.wrappers import FlattenObservation # Ensure this is imported
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 or gymnasium.wrappers not found. Evaluation script will not work.")
    # Define dummy DQN for parsing if needed, or exit early in main
    class DQN:
        @staticmethod
        def load(path, device='auto'):
            raise NotImplementedError("SB3 DQN not available.")
    class FlattenObservation: # Dummy for when SB3 is not available
        def __init__(self, env):
            self.env = env
            self.observation_space = env.observation_space # Placeholder
        def observation(self, obs): return obs # Placeholder
        def __getattr__(self, name): return getattr(self.env, name) # Pass through other attributes


from dnd_gym_env import DnDCombatEnv, Creature # Assuming Creature might be useful for type hints or direct use
# Bestiary imports will be dynamic based on args

# Helper function to be added in evaluate_models.py
def load_stats_from_module(module_path_str: str) -> dict:
    try:
        module_name_part = module_path_str.split('.')[-1] # e.g., "wolf" from "bestiary.wolf"
        # Correctly form the function name, e.g. get_wolf_stats not get_bestiary.wolf_stats
        if module_path_str.startswith("bestiary."): # Common case
             expected_func_name = f"get_{module_name_part}_stats"
        else: # More generic, assumes last part is the creature name
            expected_func_name = f"get_{module_name_part}_stats"

        module = importlib.import_module(module_path_str)
        stats_func = getattr(module, expected_func_name)
        print(f"Successfully loaded stats function '{expected_func_name}' from '{module_path_str}'")
        return stats_func()
    except ImportError:
        print(f"Error: Could not import module '{module_path_str}'. Please check the path.")
        raise
    except AttributeError:
        print(f"Error: Could not find function '{expected_func_name}' in module '{module_path_str}'.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading stats from '{module_path_str}': {e}")
        raise

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate two D&D combat AI models against each other.")
    parser.add_argument("--model1_path", type=str, required=True,
                        help="Path to the pre-trained model file for Agent 1 (plays as 'agent').")
    parser.add_argument("--model2_path", type=str, required=True,
                        help="Path to the pre-trained model file for Agent 2 (plays as 'enemy').")
    parser.add_argument("--agent_stats_module", type=str, required=True,
                        help="Python module path for Agent 1's stats (e.g., 'bestiary.wolf'). Must contain get_stats().")
    parser.add_argument("--enemy_stats_module", type=str, required=True,
                        help="Python module path for Agent 2's stats (e.g., 'bestiary.commoner'). Must contain get_stats().")
    parser.add_argument("--num_episodes", type=int, default=10000,
                        help="Number of episodes to run for evaluation.")
    parser.add_argument("--map_width", type=int, default=10,
                        help="Environment map width.")
    parser.add_argument("--map_height", type=int, default=10,
                        help="Environment map height.")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Max steps per episode before truncation.")
    # Add a seed argument for reproducibility of starting positions if desired
    parser.add_argument("--env_seed", type=int, default=None,
                        help="Seed for the environment's random number generator.")


    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    if not SB3_AVAILABLE:
        print("Stable Baselines3 is required to run this evaluation script. Please install it.")
        return

    print("Starting D&D Model Evaluation Script")
    print("------------------------------------")
    print(f"Model 1 (Agent): {args.model1_path}")
    print(f"Model 2 (Enemy): {args.model2_path}")
    print(f"Agent Stats: {args.agent_stats_module}")
    print(f"Enemy Stats: {args.enemy_stats_module}")
    print(f"Number of Episodes: {args.num_episodes}")
    print(f"Map Size: {args.map_width}x{args.map_height}")
    print(f"Max Steps per Episode: {args.max_steps}")
    if args.env_seed is not None:
        print(f"Environment Seed: {args.env_seed}")
    print("------------------------------------")

    # Load stats
    print("Loading creature stats...")
    try:
        agent_stats_data = load_stats_from_module(args.agent_stats_module)
        enemy_stats_data = load_stats_from_module(args.enemy_stats_module)
        print("Creature stats loaded successfully.")
    except Exception:
        print("Failed to load creature stats. Exiting.")
        return

    # Initialize D&D Combat Environment
    print("Initializing D&D Combat Environment...")
    env = DnDCombatEnv(
        map_width=args.map_width,
        map_height=args.map_height,
        agent_stats=agent_stats_data,
        enemy_stats=enemy_stats_data,
        grid_size=1, # Assuming default grid_size
        render_mode=None, # No rendering during evaluation
        export_frames_path=None,
        enemy_model_path=args.model2_path # Pass Model 2 path here
    )
    env.max_episode_steps = args.max_steps # Set max steps for truncation
    # Seed the environment if a seed is provided
    if args.env_seed is not None:
        # Note: Environment seeding in Gymnasium is typically done by passing seed to reset()
        # For SB3, make_vec_env handles seeding. If using direct env, seed on reset.
        # We'll ensure reset uses the seed later in the evaluation loop.
        print(f"Environment will be seeded with: {args.env_seed} during reset.")

    # Wrap the environment for Model 1
    env = FlattenObservation(env)
    print("Environment initialized and wrapped with FlattenObservation.")

    # Load Model 1 (Agent)
    print(f"Loading Model 1 (Agent) from: {args.model1_path}")
    try:
        model1 = DQN.load(args.model1_path, device='auto')
        print("Model 1 loaded successfully.")
    except Exception as e:
        print(f"Error loading Model 1 from {args.model1_path}: {e}")
        return

    model1_wins = 0
    model2_wins = 0 # Represents wins by the 'enemy' model (Model 2)
    draws = 0

    start_time = time.time()
    print(f"Starting evaluation for {args.num_episodes} episodes...")

    for episode_num in range(1, args.num_episodes + 1):
        current_episode_seed = None
        if args.env_seed is not None:
            current_episode_seed = args.env_seed + episode_num - 1
            obs, info = env.reset(seed=current_episode_seed)
        else:
            obs, info = env.reset()

        terminated = False
        truncated = False
        episode_step_count = 0

        # Ensure enemy model is loaded if specified (env handles this if enemy_model_path was given)
        # Accessing env.env since 'env' is FlattenObservation wrapped
        if not env.env.enemy_is_model_controlled and args.model2_path:
             print("Warning: Enemy model path was provided but enemy is not model-controlled in env.")


        while not terminated and not truncated:
            action, _ = model1.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_step_count += 1

        # Determine outcome
        # Access actual environment through env.env to get agent/enemy attributes
        if terminated:
            if env.env.agent.is_alive and not env.env.enemy.is_alive:
                model1_wins += 1
            elif not env.env.agent.is_alive and env.env.enemy.is_alive:
                model2_wins += 1
            else: # Both died in the same step, or some other terminal state
                draws += 1
                # print(f"Debug: Episode {episode_num} ended in a draw (terminated). Agent alive: {env.env.agent.is_alive}, Enemy alive: {env.env.enemy.is_alive}")
        elif truncated: # Episode ended due to max_steps
            draws += 1
            # print(f"Debug: Episode {episode_num} ended in a draw (truncated).")

        # Print progress
        if episode_num % 100 == 0 or episode_num == args.num_episodes:
            elapsed_time = time.time() - start_time
            print(f"Episode {episode_num}/{args.num_episodes} completed. "
                  f"M1 Wins: {model1_wins}, M2 Wins: {model2_wins}, Draws: {draws}. "
                  f"Seed: {current_episode_seed if current_episode_seed is not None else 'N/A'}. "
                  f"Steps: {episode_step_count}. "
                  f"Time: {elapsed_time:.2f}s")

    # Final Results Reporting (Placeholder)
    total_time = time.time() - start_time
    print(f"\nEvaluation finished in {total_time:.2f} seconds.")
    print("------------------------------------")
    print("           Evaluation Results       ")
    print("------------------------------------")

    if args.num_episodes > 0:
        model1_win_rate = (model1_wins / args.num_episodes) * 100
        model2_win_rate = (model2_wins / args.num_episodes) * 100
        draw_rate = (draws / args.num_episodes) * 100

        print(f"Total Episodes: {args.num_episodes}")
        print(f"Model 1 ('agent') Wins: {model1_wins} ({model1_win_rate:.2f}%)")
        print(f"Model 2 ('enemy') Wins: {model2_wins} ({model2_win_rate:.2f}%)")
        print(f"Draws: {draws} ({draw_rate:.2f}%)")
    else:
        print("No episodes were run.")

    print("------------------------------------")

if __name__ == "__main__":
    main()
