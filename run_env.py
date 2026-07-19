"""Watch two combatants fight, one micro-action per rendered frame.

Unlike training (where env.step() runs one agent micro-action plus the
opponent's entire turn internally), this viewer drives BOTH sides itself at
micro-action granularity via the env's spectator API. Each side is treated
identically — Agent 1 and Agent 2 — and every micro-action (including each of
the opponent's moves) produces its own frame and log line.

Each side is model-controlled if a model path is given and loads, otherwise it
acts randomly (sampling from the legal-action mask).
"""
import os
import argparse
from typing import Optional

import numpy as np
from gymnasium.wrappers import FlattenObservation
from gymnasium.utils.env_checker import check_env

import bestiary
from dnd_gym_env import DnDCombatEnv, load_model_any, MASKABLE_AVAILABLE
try:
    from sb3_contrib import MaskablePPO
except ImportError:
    class MaskablePPO: pass

CREATURES = ["bandit", "cat", "commoner", "goblin", "guard", "scout", "wolf"]


def get_stats(creature_name: str) -> dict:
    return getattr(bestiary, f"get_{creature_name}_stats")()


def load_side(model_path: Optional[str], label: str):
    """Return (model_or_None, uses_masks). None means a random policy."""
    if not model_path:
        print(f"{label}: random policy")
        return None, False
    if not os.path.exists(model_path):
        print(f"{label}: model path '{model_path}' not found -> random policy")
        return None, False
    try:
        model = load_model_any(model_path)
    except Exception as e:
        print(f"{label}: failed to load model ({e}) -> random policy")
        return None, False
    uses_masks = MASKABLE_AVAILABLE and isinstance(model, MaskablePPO)
    print(f"{label}: model {model_path} ({type(model).__name__}"
          f"{', masked' if uses_masks else ''})")
    return model, uses_masks


def main(agent1_creature: str, agent2_creature: str,
         agent1_model_path: Optional[str], agent2_model_path: Optional[str],
         export_frames_path: Optional[str], seed: int, max_rounds: int):
    print(f"\n--- Watching D&D Combat ---")
    print(f"Matchup: Agent 1 ({agent1_creature}) vs Agent 2 ({agent2_creature})")

    env = DnDCombatEnv(
        map_width=10, map_height=10,
        agent_stats=get_stats(agent1_creature),   # Agent 1 fills the 'agent' seat
        enemy_stats=get_stats(agent2_creature),   # Agent 2 fills the 'enemy' seat
        render_mode='human',
        export_frames_path=export_frames_path,
        enemy_model_path=None,                     # both sides are driven here, not internally
    )

    try:
        print("\nRunning Environment Checker...")
        env.export_frames_path = None  # don't export the checker's throwaway frames
        check_env(env)
        env.export_frames_path = export_frames_path
        print("Environment check passed successfully!")
    except Exception as e:
        print(f"Environment check failed: {e}")
        env.close()
        return

    flatten = FlattenObservation(env)  # same obs schema for both sides
    model1, masks1 = load_side(agent1_model_path, "Agent 1")
    model2, masks2 = load_side(agent2_model_path, "Agent 2")
    sides = {
        'agent': ("Agent 1", model1, masks1),
        'enemy': ("Agent 2", model2, masks2),
    }

    rng = np.random.default_rng(seed)

    def choose_action(prefix: str) -> int:
        _, model, uses_masks = sides[prefix]
        mask = env.action_mask_for(prefix)
        if model is None:
            return int(rng.choice(np.flatnonzero(mask)))
        flat_obs = flatten.observation(env.observation_for(prefix))
        if uses_masks:
            action, _ = model.predict(flat_obs, deterministic=True, action_masks=mask)
        else:
            action, _ = model.predict(flat_obs, deterministic=True)
        return int(np.asarray(action).item())

    obs, info = env.reset(seed=seed)
    env.event_log.clear()  # drop check_env's throwaway steps from the panel
    env.render()
    events_printed = 0

    def flush_events():
        nonlocal events_printed
        for line in list(env.event_log)[events_printed:]:
            print(f"  {line}")
        events_printed = len(env.event_log)
        print(f"    Agent 1 ({env.agent.display_name}) "
              f"HP {env.agent.current_hp}/{env.agent.max_hp} @ {env.agent.position} | "
              f"Agent 2 ({env.enemy.display_name}) "
              f"HP {env.enemy.current_hp}/{env.enemy.max_hp} @ {env.enemy.position}")

    terminated = False
    for round_num in range(max_rounds):
        for prefix in ('agent', 'enemy'):  # Agent 1 acts, then Agent 2
            actor = env._side(prefix)[0]
            if not actor.can_act():
                continue
            env.begin_turn(prefix)
            for _ in range(env._max_micro_steps(actor)):
                action_id = choose_action(prefix)
                terminated, turn_over, _ = env.spectator_step(prefix, action_id)
                env.render()
                flush_events()
                if terminated or turn_over:
                    break
            if terminated:
                break
        if terminated:
            break

    print("\nEpisode finished!")
    if terminated:
        if env.agent.is_alive and not env.enemy.is_alive:
            print(f"Result: Agent 1 ({env.agent.display_name}) won!")
        elif not env.agent.is_alive and env.enemy.is_alive:
            print(f"Result: Agent 2 ({env.enemy.display_name}) won!")
        else:
            print("Result: Both down — draw.")
    else:
        print(f"Result: Draw (reached max {max_rounds} rounds).")

    env.close()
    print("\nEnvironment closed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Watch two D&D combatants fight (symmetric spectator view)")
    parser.add_argument("--agent1_creature", type=str, default="bandit", choices=CREATURES,
                        help="Bestiary creature for Agent 1. Default: bandit")
    parser.add_argument("--agent2_creature", type=str, default="wolf", choices=CREATURES,
                        help="Bestiary creature for Agent 2. Default: wolf")
    parser.add_argument("--agent1_model", type=str, default=None,
                        help="Path to Agent 1's model. If omitted/not found, Agent 1 acts randomly.")
    parser.add_argument("--agent2_model", type=str, default=None,
                        help="Path to Agent 2's model. If omitted/not found, Agent 2 acts randomly.")
    parser.add_argument("--export_frames_path", type=str, default=None,
                        help="Directory to save a PNG per micro-action. If unset, frames are not saved.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for placement and random policies.")
    parser.add_argument("--max_rounds", type=int, default=60,
                        help="Max full rounds (Agent 1 turn + Agent 2 turn) before declaring a draw.")
    args = parser.parse_args()

    main(
        agent1_creature=args.agent1_creature,
        agent2_creature=args.agent2_creature,
        agent1_model_path=args.agent1_model,
        agent2_model_path=args.agent2_model,
        export_frames_path=args.export_frames_path,
        seed=args.seed,
        max_rounds=args.max_rounds,
    )
