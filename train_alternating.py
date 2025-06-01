import os
import argparse
from typing import Optional, Dict, Any # For type hints

# Attempt to import DRL components, handle if not available (for environments that can't install them)
try:
    from stable_baselines3 import DQN
    # MlpPolicy is generally the default for DQN if not specified, or can be imported if needed for policy_kwargs
    # from stable_baselines3.common.policies import MlpPolicy 
    from stable_baselines3.common.vec_env import DummyVecEnv
    from gymnasium.wrappers import FlattenObservation # Gymnasium for wrappers
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: Stable Baselines3 or Gymnasium components not found. Some features will be disabled.")
    # Define dummy classes if needed for type hinting or basic script structure to parse
    class DQN: pass
    # class MlpPolicy: pass # Not strictly needed if using default string "MlpPolicy"
    class DummyVecEnv: pass
    class FlattenObservation: pass

import importlib # Added for dynamic stat loading

# Project-specific imports
try:
    from dnd_gym_env import DnDCombatEnv
    # Direct imports of get_commoner_stats and get_wolf_stats are no longer needed here
    # as stats will be loaded dynamically.
    PROJECT_ENV_AVAILABLE = True
except ImportError:
    PROJECT_ENV_AVAILABLE = False
    print("Warning: Custom D&D environment or bestiary not found. Script cannot run as intended.")
    # Define dummy classes/functions if needed for script structure if env is missing
    class DnDCombatEnv: pass 
    # No need for dummy get_stats functions here anymore


# --- Global Constants and Configuration ---
# Global AGENT_STATS and ENEMY_STATS are removed. They will be loaded dynamically.

DEFAULT_MODEL_A_PATH = "models/dnd_agent1_selfplay.zip"
DEFAULT_MODEL_B_PATH = "models/dnd_agent2_selfplay.zip"

MAP_WIDTH = 10
MAP_HEIGHT = 10
GRID_SIZE = 1 # Assuming 1 unit = 1 cell for speed/range calculations in env

LOG_DIR_BASE = "./dnd_selfplay_tensorboard/"
# Ensure base log directory exists
os.makedirs(LOG_DIR_BASE, exist_ok=True)


# --- Main Script Logic ---
if __name__ == '__main__':
    if not SB3_AVAILABLE or not PROJECT_ENV_AVAILABLE:
        print("Critical components (Stable Baselines3 or D&D Environment) are missing. Exiting.")
        exit()

    parser = argparse.ArgumentParser(description="Alternating Self-Play Training for D&D Combat Agents")
    
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of full training iterations (A trains, then B trains). Default: 10")
    parser.add_argument("--timesteps_per_iteration", type=int, default=50000, 
                        help="Timesteps for each agent's training phase per iteration. Default: 50000")

    parser.add_argument("--agent1_stats_module", type=str, required=True,
                        help="Python module path for Agent 1's stats (e.g., 'bestiary.wolf'). Must contain get_<name>_stats().")
    parser.add_argument("--agent2_stats_module", type=str, required=True,
                        help="Python module path for Agent 2's stats (e.g., 'bestiary.commoner'). Must contain get_<name>_stats().")
    
    parser.add_argument("--model_a_path", type=str, default=DEFAULT_MODEL_A_PATH,
                        help=f"Path to load/save Agent 1 model. Default: {DEFAULT_MODEL_A_PATH}")
    parser.add_argument("--model_b_path", type=str, default=DEFAULT_MODEL_B_PATH,
                        help=f"Path to load/save Agent 2 model. Default: {DEFAULT_MODEL_B_PATH}")
    
    # Optional: DQN Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for DQN. Default: 0.0001")
    parser.add_argument("--buffer_size", type=int, default=50000, help="Replay buffer size for DQN. Default: 50000")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DQN. Default: 32")
    parser.add_argument("--learning_starts", type=int, default=1000, help="How many steps of random actions before learning starts. Default: 1000")

    args = parser.parse_args()

    print("--- Self-Play Training Configuration ---")
    print(f"Total Iterations: {args.iterations}")
    print(f"Timesteps per Agent per Iteration: {args.timesteps_per_iteration}")
    print(f"Agent 1 Model: {args.model_a_path}")
    print(f"Agent 2 Model: {args.model_b_path}")
    print(f"Agent 1 Stats Module: {args.agent1_stats_module}")
    print(f"Agent 2 Stats Module: {args.agent2_stats_module}")
    print(f"Learning Rate: {args.lr}")
    print(f"Buffer Size: {args.buffer_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Starts: {args.learning_starts}")
    print("--------------------------------------")

    # --- DQN Hyperparameters ---
    dqn_hyperparameters = {
        "learning_rate": args.lr,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "tau": 1.0, 
        "gamma": 0.99, 
        "train_freq": 4, 
        "gradient_steps": 1, 
        "policy_kwargs": None 
    }
    print(f"DQN Hyperparameters: {dqn_hyperparameters}")

    # --- Assume initialize_model and make_training_env will be defined in a later step ---
    # For now, to make the script runnable up to this point if those functions are missing,
    # we can define placeholder/dummy versions if SB3_AVAILABLE is True.
    # This is just for structural completeness of this subtask.
    # Ensure typing imports are present if not already
    # from typing import Optional, Dict, Any (already at the top)

    def load_stats_from_module(module_path_str: str) -> dict:
        try:
            actual_module_name = module_path_str.split('.')[-1]
            expected_func_name = f"get_{actual_module_name}_stats"

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

    print("\nLoading creature stats...")
    try:
        agent1_stats_data = load_stats_from_module(args.agent1_stats_module)
        agent2_stats_data = load_stats_from_module(args.agent2_stats_module)
        print("Creature stats loaded successfully.")
        # print(f"Agent 1 Stats: {agent1_stats_data}") # Optional: for debugging
        # print(f"Agent 2 Stats: {agent2_stats_data}") # Optional: for debugging
    except Exception as e:
        print(f"Failed to load creature stats. Exiting. Error: {e}")
        exit()
    
    # Placeholder for initialize_model function
    def initialize_model(model_path: str,
                         is_agent_a_being_initialized: bool,
                         dqn_hparams: Dict[str, Any],
                         initial_opponent_model_path: Optional[str],
                         training_env_lambda: Any) -> DQN: # Return type is SB3 DQN model
        agent_name_log = "Agent 1" if is_agent_a_being_initialized else "Agent 2"
        print(f"Initializing model for {agent_name_log} from/to {model_path}")
        print(f"  Initial opponent model for this agent: {initial_opponent_model_path}")

        if os.path.exists(model_path):
            print(f"  Loading existing model from {model_path}")
            # The env parameter in load is for updating model's env, not for creating new one.
            # If custom objects are involved, it might be needed. For DQN, often not for basic load.
            return DQN.load(model_path, device='auto')
        else:
            print(f"  Creating new model, to be saved at {model_path}")
            if not SB3_AVAILABLE:
                raise RuntimeError("Stable Baselines3 is not available, cannot create new model.")
            
            # The training_env_lambda should return a ready-to-use environment (already wrapped)
            # However, DQN usually expects a VecEnv.
            print("  Creating environment for new model initialization using training_env_lambda...")
            base_env_for_init = training_env_lambda() # This should be FlattenObservation(DnDCombatEnv(...))
            vec_env_for_init = DummyVecEnv([lambda: base_env_for_init]) # Wrap it for DQN
            
            print(f"  New model parameters: policy=MlpPolicy, env_type={type(vec_env_for_init)}")
            model = DQN(
                "MlpPolicy",
                vec_env_for_init,
                tensorboard_log=None, # Will be set before learning
                **dqn_hparams
            )
            # vec_env_for_init.close() # Close after model init if not used by caller
            return model

    # Refined make_training_env function
    def make_training_env(is_training_agent_a: bool,
                          opponent_model_path: Optional[str],
                          current_iteration: int,
                          training_agent_name: str,
                          agent1_stats: Dict[str, Any],
                          agent2_stats: Dict[str, Any]) -> DnDCombatEnv: # Returns base env, will be wrapped by DummyVecEnv

        print(f"Creating training env for: {training_agent_name} (Iteration {current_iteration})")
        print(f"  - Agent being trained is: {'Agent 1' if is_training_agent_a else 'Agent 2'}")
        print(f"  - Opponent model path: {opponent_model_path}")

        if is_training_agent_a:
            current_agent_true_stats = agent1_stats
            current_opponent_true_stats = agent2_stats
            print(f"  - Env Agent Stats (Agent 1): AC {current_agent_true_stats.get('ac', 'N/A')}, HP {current_agent_true_stats.get('max_hp', 'N/A')}")
            print(f"  - Env Enemy Stats (Agent 2): AC {current_opponent_true_stats.get('ac', 'N/A')}, HP {current_opponent_true_stats.get('max_hp', 'N/A')}")
        else:
            current_agent_true_stats = agent2_stats
            current_opponent_true_stats = agent1_stats
            print(f"  - Env Agent Stats (Agent 2): AC {current_agent_true_stats.get('ac', 'N/A')}, HP {current_agent_true_stats.get('max_hp', 'N/A')}")
            print(f"  - Env Enemy Stats (Agent 1): AC {current_opponent_true_stats.get('ac', 'N/A')}, HP {current_opponent_true_stats.get('max_hp', 'N/A')}")

        if not PROJECT_ENV_AVAILABLE:
            raise RuntimeError("DnDCombatEnv not available for make_training_env.")

        # The environment's "agent" is the one being trained.
        # Its "enemy" is the opponent, potentially model-controlled by opponent_model_path.
        env = DnDCombatEnv(
            map_width=MAP_WIDTH,
            map_height=MAP_HEIGHT,
            agent_stats=current_agent_true_stats,
            enemy_stats=current_opponent_true_stats,
            enemy_model_path=opponent_model_path,
            render_mode=None
        )
        # Return the base environment. FlattenObservation and DummyVecEnv will be applied outside or in initialize_model.
        return env # Return the base env

    # --- Initialize Models ---
    # Lambdas for training_env_lambda should now return the base DnDCombatEnv,
    # which will be wrapped by FlattenObservation then DummyVecEnv inside initialize_model or before .learn()

    print(f"\nInitializing Agent 1 ({args.agent1_stats_module.split('.')[-1]})...")
    training_env_lambda_a_init = lambda: FlattenObservation(make_training_env(
        is_training_agent_a=True,
        opponent_model_path=(args.model_b_path if os.path.exists(args.model_b_path) else None),
        current_iteration=0,
        training_agent_name="Init_Agent1_vs_Agent2",
        agent1_stats=agent1_stats_data,
        agent2_stats=agent2_stats_data
    ))
    model_a = initialize_model(
        model_path=args.model_a_path,
        is_agent_a_being_initialized=True, 
        dqn_hparams=dqn_hyperparameters,
        initial_opponent_model_path=args.model_b_path if os.path.exists(args.model_b_path) else None,
        training_env_lambda=training_env_lambda_a_init
    )

    print(f"\nInitializing Agent 2 ({args.agent2_stats_module.split('.')[-1]})...")
    training_env_lambda_b_init = lambda: FlattenObservation(make_training_env(
        is_training_agent_a=False,
        opponent_model_path=(args.model_a_path if os.path.exists(args.model_a_path) else None),
        current_iteration=0,
        training_agent_name="Init_Agent2_vs_Agent1",
        agent1_stats=agent1_stats_data,
        agent2_stats=agent2_stats_data
    ))
    model_b = initialize_model(
        model_path=args.model_b_path,
        is_agent_a_being_initialized=False, 
        dqn_hparams=dqn_hyperparameters,
        initial_opponent_model_path=args.model_a_path if os.path.exists(args.model_a_path) else None,
        training_env_lambda=training_env_lambda_b_init
    )

    # --- Main Alternating Training Loop ---
    print(f"\n--- Starting Alternating Training for {args.iterations} Iterations ---")
    for i in range(args.iterations):
        iteration_num = i + 1
        print(f"\n===== Iteration {iteration_num}/{args.iterations} =====")

        # --- Train Agent A (Agent 1) against current Agent B (Agent 2) ---
        agent_a_creature_name = args.agent1_stats_module.split('.')[-1]
        agent_a_log_name = f"Agent1_{agent_a_creature_name}_Iter{iteration_num}"
        print(f"\nTraining {agent_a_log_name} (Agent 1) against current Agent 2 ({args.model_b_path})...")
        
        opponent_b_model_to_use = args.model_b_path if os.path.exists(args.model_b_path) else None
        if opponent_b_model_to_use:
             print(f"Agent 1 will train against loaded Agent 2 model: {opponent_b_model_to_use}")
        else:
             print(f"Agent 1 will train against rule-based Agent 2 ({args.agent2_stats_module.split('.')[-1]}) as {args.model_b_path} not found or not yet trained.")

        # Lambda now returns the base environment, to be wrapped
        env_a_base_lambda = lambda: make_training_env(
            is_training_agent_a=True, 
            opponent_model_path=opponent_b_model_to_use,
            current_iteration=iteration_num,
            training_agent_name=agent_a_log_name,
            agent1_stats=agent1_stats_data,
            agent2_stats=agent2_stats_data
        )
        vec_env_a = DummyVecEnv([lambda: FlattenObservation(env_a_base_lambda())])
        
        model_a.set_env(vec_env_a)
        model_a.tensorboard_log = os.path.join(LOG_DIR_BASE, agent_a_log_name)
        os.makedirs(model_a.tensorboard_log, exist_ok=True)

        print(f"Agent A ({agent_a_log_name}) learning for {args.timesteps_per_iteration} timesteps. Logs: {model_a.tensorboard_log}")
        model_a.learn(
            total_timesteps=args.timesteps_per_iteration,
            reset_num_timesteps=False # Continue learning on the same model
        )
        model_a.save(args.model_a_path)
        print(f"{agent_a_log_name} trained and saved to {args.model_a_path}")
        vec_env_a.close()

        # --- Train Agent B (Agent 2) against updated Agent A (Agent 1) ---
        agent_b_creature_name = args.agent2_stats_module.split('.')[-1]
        agent_b_log_name = f"Agent2_{agent_b_creature_name}_Iter{iteration_num}"
        print(f"\nTraining {agent_b_log_name} (Agent 2) against updated Agent 1 ({args.model_a_path})...")

        opponent_a_model_to_use = args.model_a_path # Agent 1's model should exist now
        print(f"Agent 2 will train against loaded Agent 1 model: {opponent_a_model_to_use}")

        env_b_base_lambda = lambda: make_training_env(
            is_training_agent_a=False, 
            opponent_model_path=opponent_a_model_to_use,
            current_iteration=iteration_num,
            training_agent_name=agent_b_log_name,
            agent1_stats=agent1_stats_data,
            agent2_stats=agent2_stats_data
        )
        vec_env_b = DummyVecEnv([lambda: FlattenObservation(env_b_base_lambda())])

        model_b.set_env(vec_env_b)
        model_b.tensorboard_log = os.path.join(LOG_DIR_BASE, agent_b_log_name)
        os.makedirs(model_b.tensorboard_log, exist_ok=True)
        
        print(f"Agent B ({agent_b_log_name}) learning for {args.timesteps_per_iteration} timesteps. Logs: {model_b.tensorboard_log}")
        model_b.learn(
            total_timesteps=args.timesteps_per_iteration,
            reset_num_timesteps=False # Continue learning
        )
        model_b.save(args.model_b_path)
        print(f"{agent_b_log_name} trained and saved to {args.model_b_path}")
        vec_env_b.close()

    print("\n--- Alternating Self-Play Training Complete ---")
    print(f"Final Agent 1 ({args.agent1_stats_module.split('.')[-1]}) model saved to: {args.model_a_path}")
    print(f"Final Agent 2 ({args.agent2_stats_module.split('.')[-1]}) model saved to: {args.model_b_path}")
    print(f"TensorBoard logs saved in directory: {os.path.abspath(LOG_DIR_BASE)}")
    print(f"To view logs: tensorboard --logdir=\"{os.path.abspath(LOG_DIR_BASE)}\"")