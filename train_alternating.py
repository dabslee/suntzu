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

# Project-specific imports
try:
    from dnd_gym_env import DnDCombatEnv
    from bestiary.commoner import get_commoner_stats
    from bestiary.wolf import get_wolf_stats
    PROJECT_ENV_AVAILABLE = True
except ImportError:
    PROJECT_ENV_AVAILABLE = False
    print("Warning: Custom D&D environment or bestiary not found. Script cannot run as intended.")
    # Define dummy classes/functions if needed for script structure if env is missing
    class DnDCombatEnv: pass 
    def get_commoner_stats(): return {}
    def get_wolf_stats(): return {}


# --- Global Constants and Configuration ---
# These will be loaded only if the environment itself was loaded
AGENT_STATS = get_commoner_stats() if PROJECT_ENV_AVAILABLE else {}
ENEMY_STATS = get_wolf_stats() if PROJECT_ENV_AVAILABLE else {}

DEFAULT_MODEL_A_PATH = "models/dnd_commoner_selfplay_agent.zip"  # Agent A = Commoner
DEFAULT_MODEL_B_PATH = "models/dnd_wolf_selfplay_agent.zip"      # Agent B = Wolf

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
    
    parser.add_argument("--model_a_path", type=str, default=DEFAULT_MODEL_A_PATH,
                        help=f"Path to load/save Agent A (Commoner) model. Default: {DEFAULT_MODEL_A_PATH}")
    parser.add_argument("--model_b_path", type=str, default=DEFAULT_MODEL_B_PATH,
                        help=f"Path to load/save Agent B (Wolf) model. Default: {DEFAULT_MODEL_B_PATH}")
    
    # Optional: DQN Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for DQN. Default: 0.0001")
    parser.add_argument("--buffer_size", type=int, default=50000, help="Replay buffer size for DQN. Default: 50000")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DQN. Default: 32")
    parser.add_argument("--learning_starts", type=int, default=1000, help="How many steps of random actions before learning starts. Default: 1000")

    args = parser.parse_args()

    print("--- Self-Play Training Configuration ---")
    print(f"Total Iterations: {args.iterations}")
    print(f"Timesteps per Agent per Iteration: {args.timesteps_per_iteration}")
    print(f"Agent A (Commoner) Model: {args.model_a_path}")
    print(f"Agent B (Wolf) Model: {args.model_b_path}")
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
    
    # Placeholder for initialize_model function (to be implemented later)
    def initialize_model(model_path, is_agent_a_being_initialized, dqn_hparams, initial_opponent_model_path, training_env_lambda):
        print(f"Placeholder: Would initialize model for {'Agent A' if is_agent_a_being_initialized else 'Agent B'} from/to {model_path}")
        print(f"Placeholder: Initial opponent model: {initial_opponent_model_path}")
        if os.path.exists(model_path):
            print(f"Placeholder: Loading model from {model_path}")
            # return DQN.load(model_path, env=training_env_lambda()) # Env might be needed if custom policy
            return DQN.load(model_path) # Simpler load
        else:
            print(f"Placeholder: Creating new model, to be saved at {model_path}")
            # env = training_env_lambda() # Create an env for initialization
            # model = DQN("MlpPolicy", env, tensorboard_log=None, **dqn_hparams)
            # env.close()
            # For this placeholder, we can't create a real model without a real env.
            # Returning a mock/dummy object or None if SB3 is available but env setup is complex.
            # For now, let's assume it would return a new DQN model.
            # This part will be fully implemented when make_training_env is also implemented.
            print("Warning: initialize_model is a placeholder. Actual model creation needs make_training_env.")
            # We need a dummy VecEnv to initialize DQN if no model path exists
            # This is tricky as make_training_env is also a placeholder for now.
            # For the script to run without erroring here, we'll return a basic DQN if SB3 is available.
            if SB3_AVAILABLE:
                 temp_env = DummyVecEnv([lambda: FlattenObservation(DnDCombatEnv(MAP_WIDTH, MAP_HEIGHT, AGENT_STATS, ENEMY_STATS))])
                 model = DQN("MlpPolicy", temp_env, **dqn_hparams)
                 temp_env.close()
                 return model
            return None # Should not be reached if SB3_AVAILABLE check passed earlier

    # Placeholder for make_training_env function (to be implemented later)
    def make_training_env(is_training_agent_a, opponent_model_path, current_iteration, training_agent_name):
        print(f"Placeholder: Would create training env for {'Agent A' if is_training_agent_a else 'Agent B'}")
        print(f"Placeholder: Opponent model: {opponent_model_path}, Iter: {current_iteration}, Name: {training_agent_name}")
        # This will be fully implemented later. For now, return a basic env for placeholder initialize_model.
        # Actual implementation will set agent_stats, enemy_stats, and enemy_model_path based on args.
        if PROJECT_ENV_AVAILABLE:
            agent_s = AGENT_STATS if is_training_agent_a else ENEMY_STATS
            enemy_s = ENEMY_STATS if is_training_agent_a else AGENT_STATS
            # If training Agent A, enemy uses opponent_model_path (which is B's model path)
            # If training Agent B, enemy uses opponent_model_path (which is A's model path)
            # This is tricky because the DnDCombatEnv's "enemy_model_path" is for its internal enemy.
            # make_training_env needs to correctly assign roles.
            
            # If Agent A is training, it's the "agent" in the env, Wolf is "enemy" (potentially model-controlled by opponent_model_path)
            # If Agent B is training, it's the "agent" in the env, Commoner is "enemy" (potentially model-controlled by opponent_model_path)
            current_agent_stats = AGENT_STATS if is_training_agent_a else ENEMY_STATS
            current_enemy_stats = ENEMY_STATS if is_training_agent_a else AGENT_STATS
            # The 'opponent_model_path' is for the 'enemy' in this setup
            
            return FlattenObservation(
                DnDCombatEnv(MAP_WIDTH, MAP_HEIGHT, current_agent_stats, current_enemy_stats, 
                             enemy_model_path=opponent_model_path, # This sets the enemy of the training agent
                             render_mode=None)
            )
        return None # Should not be reached if PROJECT_ENV_AVAILABLE check passed

    # --- Initialize Models ---
    print("\nInitializing Agent A (Commoner)...")
    # For Agent A (Commoner), its opponent is Agent B (Wolf)
    # When Agent A is initialized for the first time, Agent B might not exist as a model yet.
    model_a = initialize_model(
        model_path=args.model_a_path,
        is_agent_a_being_initialized=True, 
        dqn_hparams=dqn_hyperparameters,
        initial_opponent_model_path=args.model_b_path if os.path.exists(args.model_b_path) else None, # Use existing B if available
        training_env_lambda=lambda: make_training_env(True, (args.model_b_path if os.path.exists(args.model_b_path) else None), 0, "Init_A_vs_B")
    )

    print("\nInitializing Agent B (Wolf)...")
    # For Agent B (Wolf), its opponent is Agent A (Commoner)
    model_b = initialize_model(
        model_path=args.model_b_path,
        is_agent_a_being_initialized=False, 
        dqn_hparams=dqn_hyperparameters,
        initial_opponent_model_path=args.model_a_path if os.path.exists(args.model_a_path) else None, # Use existing A if available
        training_env_lambda=lambda: make_training_env(False, (args.model_a_path if os.path.exists(args.model_a_path) else None), 0, "Init_B_vs_A")
    )

    # --- Main Alternating Training Loop ---
    print(f"\n--- Starting Alternating Training for {args.iterations} Iterations ---")
    for i in range(args.iterations):
        iteration_num = i + 1
        print(f"\n===== Iteration {iteration_num}/{args.iterations} =====")

        # --- Train Agent A (Commoner) against current Agent B ---
        agent_a_log_name = f"AgentA_Commoner_Iter{iteration_num}"
        print(f"\nTraining {agent_a_log_name} (Agent A) against current Agent B ({args.model_b_path})...")
        
        opponent_b_model_to_use = args.model_b_path if os.path.exists(args.model_b_path) else None
        if opponent_b_model_to_use:
             print(f"Agent A will train against loaded Agent B model: {opponent_b_model_to_use}")
        else:
             print(f"Agent A will train against rule-based Agent B (Wolf) as {args.model_b_path} not found or not yet trained.")

        env_a_lambda = lambda: make_training_env(
            is_training_agent_a=True, 
            opponent_model_path=opponent_b_model_to_use,
            current_iteration=iteration_num,
            training_agent_name=agent_a_log_name
        )
        vec_env_a = DummyVecEnv([env_a_lambda])
        
        model_a.set_env(vec_env_a) # Use vec_env_a now
        # Ensure tensorboard_log path is correctly formatted and unique per agent and iteration
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

        # --- Train Agent B (Wolf) against updated Agent A ---
        agent_b_log_name = f"AgentB_Wolf_Iter{iteration_num}"
        print(f"\nTraining {agent_b_log_name} (Agent B) against updated Agent A ({args.model_a_path})...")

        opponent_a_model_to_use = args.model_a_path # Agent A's model should exist now
        print(f"Agent B will train against loaded Agent A model: {opponent_a_model_to_use}")

        env_b_lambda = lambda: make_training_env(
            is_training_agent_a=False, 
            opponent_model_path=opponent_a_model_to_use,
            current_iteration=iteration_num,
            training_agent_name=agent_b_log_name
        )
        vec_env_b = DummyVecEnv([env_b_lambda])

        model_b.set_env(vec_env_b) # Use vec_env_b now
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
    print(f"Final Agent A (Commoner) model saved to: {args.model_a_path}")
    print(f"Final Agent B (Wolf) model saved to: {args.model_b_path}")
    print(f"TensorBoard logs saved in directory: {os.path.abspath(LOG_DIR_BASE)}")
    print(f"To view logs: tensorboard --logdir=\"{os.path.abspath(LOG_DIR_BASE)}\"")