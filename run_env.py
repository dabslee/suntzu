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

from bestiary import CREATURES, get_stats
from sunztu import spectator
from sunztu.env import DnDCombatEnv


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
    policies = {
        'agent': spectator.load_policy(agent1_model_path, "Agent 1"),
        'enemy': spectator.load_policy(agent2_model_path, "Agent 2"),
    }
    rng = np.random.default_rng(seed)

    events_printed = 0

    def on_micro_action(prefix: Optional[str]) -> None:
        """Render a frame and print the log for every micro-action, both sides."""
        nonlocal events_printed
        if prefix is None:  # opening state
            env.event_log.clear()  # drop check_env's throwaway steps from the panel
            events_printed = 0
            env.render()
            return
        env.render()
        for line in list(env.event_log)[events_printed:]:
            print(f"  {line}")
        events_printed = len(env.event_log)
        print(f"    Agent 1 ({env.agent.display_name}) "
              f"HP {env.agent.current_hp}/{env.agent.max_hp} @ {env.agent.position} | "
              f"Agent 2 ({env.enemy.display_name}) "
              f"HP {env.enemy.current_hp}/{env.enemy.max_hp} @ {env.enemy.position}")

    outcome = spectator.play_episode(
        env, flatten, policies, rng,
        max_rounds=max_rounds, seed=seed, on_micro_action=on_micro_action)

    print("\nEpisode finished!")
    if outcome == 'agent1':
        print(f"Result: Agent 1 ({env.agent.display_name}) won!")
    elif outcome == 'agent2':
        print(f"Result: Agent 2 ({env.enemy.display_name}) won!")
    elif env.agent.is_alive or env.enemy.is_alive:
        print(f"Result: Draw (reached max {max_rounds} rounds).")
    else:
        print("Result: Both down — draw.")

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
