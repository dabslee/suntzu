"""Symmetric two-agent driving of DnDCombatEnv.

`env.step()` is asymmetric by design — one agent micro-action plus the
opponent's *entire* turn resolved internally — because that is the single-agent
RL training contract. For watching a fight and for scoring two policies against
each other we want the opposite: both combatants treated identically, each
acting one micro-action at a time through the same code path.

This module owns that loop so [run_env.py](run_env.py) and
[evaluate_models.py](evaluate_models.py) share it exactly — what you watch is
what gets scored. Seats are 'agent' (Agent 1) and 'enemy' (Agent 2); a side
with no model acts randomly, sampling from its legal-action mask.
"""
import os
from typing import Dict, Optional, Tuple, Callable, Any

import numpy as np

from dnd_gym_env import load_model_any, MASKABLE_AVAILABLE

try:
    from sb3_contrib import MaskablePPO
except ImportError:  # sb3-contrib is optional
    class MaskablePPO: pass

# Env seats, in initiative order. 'agent' is Agent 1, 'enemy' is Agent 2 —
# these are seat names in the env, not a hero/monster distinction.
SIDES = ('agent', 'enemy')

# A policy is (model_or_None, uses_action_masks).
Policy = Tuple[Optional[Any], bool]


def side_label(prefix: str) -> str:
    return "Agent 1" if prefix == 'agent' else "Agent 2"


def load_policy(model_path: Optional[str], label: str, verbose: bool = True) -> Policy:
    """Load a side's policy. Returns (None, False) — a random policy — when no
    usable model is given, so a side can always be run as a baseline."""
    def say(msg: str) -> None:
        if verbose:
            print(msg)

    if not model_path:
        say(f"{label}: random policy")
        return None, False
    if not os.path.exists(model_path):
        say(f"{label}: model path '{model_path}' not found -> random policy")
        return None, False
    try:
        model = load_model_any(model_path)
    except Exception as e:
        say(f"{label}: failed to load model ({e}) -> random policy")
        return None, False
    uses_masks = MASKABLE_AVAILABLE and isinstance(model, MaskablePPO)
    say(f"{label}: model {model_path} ({type(model).__name__}"
        f"{', masked' if uses_masks else ''})")
    return model, uses_masks


def choose_action(env, prefix: str, policy: Policy, flatten, rng: np.random.Generator) -> int:
    """Pick one micro-action for `prefix` from its own perspective."""
    model, uses_masks = policy
    mask = env.action_mask_for(prefix)
    if model is None:
        return int(rng.choice(np.flatnonzero(mask)))
    flat_obs = flatten.observation(env.observation_for(prefix))
    if uses_masks:
        action, _ = model.predict(flat_obs, deterministic=True, action_masks=mask)
    else:
        action, _ = model.predict(flat_obs, deterministic=True)
    return int(np.asarray(action).item())


def play_episode(env, flatten, policies: Dict[str, Policy], rng: np.random.Generator,
                 max_rounds: int = 60, seed: Optional[int] = None,
                 on_micro_action: Optional[Callable[[Optional[str]], None]] = None) -> str:
    """Run one full fight, both sides driven a micro-action at a time.

    `on_micro_action(prefix)` is called after every micro-action (and once with
    None for the opening state) so a viewer can render a frame per action.
    Returns 'agent1', 'agent2', or 'draw'.
    """
    env.reset(seed=seed)
    if on_micro_action:
        on_micro_action(None)

    terminated = False
    for _ in range(max_rounds):
        for prefix in SIDES:
            actor = env.creature_for(prefix)
            if not actor.can_act():
                continue
            env.begin_turn(prefix)
            for _ in range(env.micro_step_limit(prefix)):
                action_id = choose_action(env, prefix, policies[prefix], flatten, rng)
                terminated, turn_over, _ = env.spectator_step(prefix, action_id)
                if on_micro_action:
                    on_micro_action(prefix)
                if terminated or turn_over:
                    break
            if terminated:
                break
        if terminated:
            break

    agent1_alive, agent2_alive = env.agent.is_alive, env.enemy.is_alive
    if terminated and agent1_alive and not agent2_alive:
        return 'agent1'
    if terminated and agent2_alive and not agent1_alive:
        return 'agent2'
    return 'draw'  # both down, or the round cap was reached
