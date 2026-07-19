"""Shared training setup for the MaskablePPO trainers.

Both train_agent.py and train_alternating.py build the same stack:
FlattenObservation envs (verbose only on the first copy) inside a DummyVecEnv
inside VecNormalize. Kept here so the two scripts can't drift apart.
"""
from typing import Any, Dict, Optional

from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sunztu.env import DnDCombatEnv

# Measured on guard vs. wolf (see README "Training performance"): masking +
# vectorized envs + reward normalization are each load-bearing.
DEFAULT_PPO_HYPERPARAMS: Dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_steps": 256,      # rollout length per env (n_envs * n_steps per update)
    "batch_size": 512,   # must divide n_envs * n_steps
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,    # entropy bonus for exploration
}


def make_training_vec_env(agent_stats: Dict[str, Any],
                          enemy_stats: Dict[str, Any],
                          opponent_model_path: Optional[str],
                          n_envs: int,
                          map_width: int = 10,
                          map_height: int = 10) -> VecNormalize:
    """N env copies in one process: MaskablePPO batches the policy forward
    pass across them, which is where the wall-clock speedup comes from.

    VecNormalize(norm_reward=True) rescales the +/-100 terminal rewards for
    PPO's value function — without it PPO plateaus at ~0% wins. Observations
    are NOT normalized, so saved policies work standalone at inference (and
    as enemy models). Reward running stats restart per vec env, which is fine
    in self-play since the opponent (and thus the reward scale) changes too.
    """
    def make_one(env_verbose: bool) -> FlattenObservation:
        env = DnDCombatEnv(
            map_width=map_width, map_height=map_height,
            agent_stats=agent_stats, enemy_stats=enemy_stats,
            enemy_model_path=opponent_model_path, render_mode=None,
            verbose=env_verbose)  # only the first env copy logs its setup
        return FlattenObservation(env)

    factories = [(lambda v=(i == 0): make_one(v)) for i in range(n_envs)]
    return VecNormalize(DummyVecEnv(factories),
                        norm_obs=False, norm_reward=True, gamma=0.99)
