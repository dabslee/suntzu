"""Alternating self-play training with MaskablePPO.

Each iteration trains Agent 1 against the latest Agent 2 checkpoint, then
Agent 2 against the updated Agent 1. Uses action masking (the env's
action_masks() hook) so the policy never samples currently-invalid actions,
and several vectorized envs per model so the policy forward pass is batched —
both give large wall-clock speedups over the old single-env DQN setup.

Note: model checkpoints are MaskablePPO; old DQN self-play checkpoints cannot
be resumed (they can still be used as *opponents* via enemy_model_path).
"""
import os
import argparse
import importlib
from typing import Optional, Dict, Any

try:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from gymnasium.wrappers import FlattenObservation
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: sb3-contrib not found. Install with: pip install sb3-contrib")

try:
    from dnd_gym_env import DnDCombatEnv
    PROJECT_ENV_AVAILABLE = True
except ImportError:
    PROJECT_ENV_AVAILABLE = False
    print("Warning: dnd_gym_env not found. Script cannot run as intended.")

DEFAULT_MODEL_A_PATH = "models/dnd_agent1_selfplay.zip"
DEFAULT_MODEL_B_PATH = "models/dnd_agent2_selfplay.zip"

MAP_WIDTH = 10
MAP_HEIGHT = 10

LOG_DIR_BASE = "./dnd_selfplay_tensorboard/"
os.makedirs(LOG_DIR_BASE, exist_ok=True)


def load_stats_from_module(module_path_str: str) -> dict:
    actual_module_name = module_path_str.split('.')[-1]
    expected_func_name = f"get_{actual_module_name}_stats"
    module = importlib.import_module(module_path_str)
    stats_func = getattr(module, expected_func_name)
    print(f"Loaded stats function '{expected_func_name}' from '{module_path_str}'")
    return stats_func()


def make_vec_env(agent_stats: Dict[str, Any], enemy_stats: Dict[str, Any],
                 opponent_model_path: Optional[str], n_envs: int) -> "DummyVecEnv":
    """N copies of the env in one process: MaskablePPO batches the policy
    forward pass across them, which is where the speedup comes from."""
    def make_one(env_verbose: bool = False):
        env = DnDCombatEnv(
            map_width=MAP_WIDTH, map_height=MAP_HEIGHT,
            agent_stats=agent_stats, enemy_stats=enemy_stats,
            enemy_model_path=opponent_model_path, render_mode=None,
            verbose=env_verbose)  # only the first env copy logs its setup
        return FlattenObservation(env)
    factories = [(lambda v=(i == 0): make_one(v)) for i in range(n_envs)]
    # VecNormalize(norm_reward=True) rescales the +/-100 terminal rewards for
    # PPO's value function — without it PPO plateaus at ~0% wins. Observations
    # are NOT normalized, so saved policies work standalone at inference (and
    # as enemy models). Reward running stats restart each iteration, which is
    # fine since the opponent (and thus the reward distribution) changes too.
    return VecNormalize(DummyVecEnv(factories),
                        norm_obs=False, norm_reward=True, gamma=0.99)


def initialize_model(model_path: str, vec_env: "DummyVecEnv",
                     hparams: Dict[str, Any]) -> "MaskablePPO":
    if os.path.exists(model_path):
        print(f"  Loading existing model from {model_path}")
        try:
            model = MaskablePPO.load(model_path, device='auto')
        except Exception as e:
            raise SystemExit(
                f"Existing checkpoint at {model_path} is not a MaskablePPO model "
                f"(old DQN self-play checkpoints are incompatible — delete/rename it "
                f"or pass a different path). Load error: {e}")
        model.set_env(vec_env)
        return model
    print(f"  Creating new MaskablePPO model (will save to {model_path})")
    return MaskablePPO("MlpPolicy", vec_env, tensorboard_log=None, **hparams)


if __name__ == '__main__':
    if not SB3_AVAILABLE or not PROJECT_ENV_AVAILABLE:
        raise SystemExit("Critical components (sb3-contrib or D&D environment) are missing.")

    parser = argparse.ArgumentParser(description="Alternating Self-Play Training for D&D Combat Agents (MaskablePPO)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of full training iterations (A trains, then B trains). Default: 10")
    parser.add_argument("--timesteps_per_iteration", type=int, default=50000,
                        help="Timesteps for each agent's training phase per iteration. Default: 50000")
    parser.add_argument("--agent1_stats_module", type=str, required=True,
                        help="Python module path for Agent 1's stats (e.g., 'bestiary.wolf').")
    parser.add_argument("--agent2_stats_module", type=str, required=True,
                        help="Python module path for Agent 2's stats (e.g., 'bestiary.goblin').")
    parser.add_argument("--model_a_path", type=str, default=DEFAULT_MODEL_A_PATH,
                        help=f"Path to load/save Agent 1 model. Default: {DEFAULT_MODEL_A_PATH}")
    parser.add_argument("--model_b_path", type=str, default=DEFAULT_MODEL_B_PATH,
                        help=f"Path to load/save Agent 2 model. Default: {DEFAULT_MODEL_B_PATH}")
    # PPO hyperparameters
    parser.add_argument("--n_envs", type=int, default=8,
                        help="Vectorized envs per model (batched inference). Default: 8")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate. Default: 3e-4")
    parser.add_argument("--n_steps", type=int, default=256,
                        help="Rollout length per env before each update. Default: 256")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Minibatch size (must divide n_envs * n_steps). Default: 512")
    parser.add_argument("--n_epochs", type=int, default=10, help="PPO epochs per update. Default: 10")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Entropy bonus (exploration). Default: 0.01")
    args = parser.parse_args()

    if (args.n_envs * args.n_steps) % args.batch_size != 0:
        raise SystemExit(f"--batch_size ({args.batch_size}) must divide "
                         f"n_envs * n_steps ({args.n_envs * args.n_steps}).")

    print("--- Self-Play Training Configuration (MaskablePPO) ---")
    print(f"Iterations: {args.iterations}, timesteps/agent/iteration: {args.timesteps_per_iteration}")
    print(f"Agent 1: {args.agent1_stats_module} -> {args.model_a_path}")
    print(f"Agent 2: {args.agent2_stats_module} -> {args.model_b_path}")
    print(f"n_envs: {args.n_envs}, lr: {args.lr}, n_steps: {args.n_steps}, "
          f"batch_size: {args.batch_size}, n_epochs: {args.n_epochs}, ent_coef: {args.ent_coef}")
    print("------------------------------------------------------")

    ppo_hyperparameters = {
        "learning_rate": args.lr,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": args.ent_coef,
    }

    print("\nLoading creature stats...")
    agent1_stats_data = load_stats_from_module(args.agent1_stats_module)
    agent2_stats_data = load_stats_from_module(args.agent2_stats_module)

    agent1_name = args.agent1_stats_module.split('.')[-1]
    agent2_name = args.agent2_stats_module.split('.')[-1]

    os.makedirs(os.path.dirname(args.model_a_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.model_b_path) or ".", exist_ok=True)

    model_a = None
    model_b = None

    print(f"\n--- Starting Alternating Training for {args.iterations} Iterations ---")
    for i in range(args.iterations):
        iteration_num = i + 1
        print(f"\n===== Iteration {iteration_num}/{args.iterations} =====")

        # --- Train Agent 1 against the current Agent 2 ---
        opponent_b = args.model_b_path if os.path.exists(args.model_b_path) else None
        print(f"\nTraining Agent 1 ({agent1_name}) vs "
              f"{'model ' + opponent_b if opponent_b else 'rule-based ' + agent2_name}...")
        vec_env_a = make_vec_env(agent1_stats_data, agent2_stats_data, opponent_b, args.n_envs)
        if model_a is None:
            model_a = initialize_model(args.model_a_path, vec_env_a, ppo_hyperparameters)
        else:
            model_a.set_env(vec_env_a)
        model_a.tensorboard_log = os.path.join(LOG_DIR_BASE, f"Agent1_{agent1_name}_Iter{iteration_num}")
        model_a.learn(total_timesteps=args.timesteps_per_iteration, reset_num_timesteps=False)
        model_a.save(args.model_a_path)
        print(f"Agent 1 saved to {args.model_a_path}")
        vec_env_a.close()

        # --- Train Agent 2 against the updated Agent 1 ---
        print(f"\nTraining Agent 2 ({agent2_name}) vs model {args.model_a_path}...")
        vec_env_b = make_vec_env(agent2_stats_data, agent1_stats_data, args.model_a_path, args.n_envs)
        if model_b is None:
            model_b = initialize_model(args.model_b_path, vec_env_b, ppo_hyperparameters)
        else:
            model_b.set_env(vec_env_b)
        model_b.tensorboard_log = os.path.join(LOG_DIR_BASE, f"Agent2_{agent2_name}_Iter{iteration_num}")
        model_b.learn(total_timesteps=args.timesteps_per_iteration, reset_num_timesteps=False)
        model_b.save(args.model_b_path)
        print(f"Agent 2 saved to {args.model_b_path}")
        vec_env_b.close()

    print("\n--- Alternating Self-Play Training Complete ---")
    print(f"Final Agent 1 ({agent1_name}) model: {args.model_a_path}")
    print(f"Final Agent 2 ({agent2_name}) model: {args.model_b_path}")
    print(f"TensorBoard logs: {os.path.abspath(LOG_DIR_BASE)}")
    print(f"To view logs: tensorboard --logdir=\"{os.path.abspath(LOG_DIR_BASE)}\"")
