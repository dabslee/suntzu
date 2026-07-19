"""Score two policies head-to-head over many episodes.

Both combatants are driven symmetrically (see [spectator.py](spectator.py)):
each acts one micro-action at a time through the same code path, so neither
side gets an advantage from how the environment happens to be wired. This is
the same loop [run_env.py](run_env.py) uses to render a fight, so what you
watch is what gets scored.

Either side may be left without a model to act randomly, which is a useful
baseline (a trained policy should comfortably beat random).
"""
import argparse
import importlib
import time
from typing import Optional

import numpy as np

try:
    from gymnasium.wrappers import FlattenObservation
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: gymnasium not found. Evaluation script will not work.")

from dnd_gym_env import DnDCombatEnv
import spectator


def load_stats_from_module(module_path_str: str) -> dict:
    """Import e.g. 'bestiary.wolf' and call its get_wolf_stats()."""
    creature = module_path_str.split('.')[-1]
    expected_func_name = f"get_{creature}_stats"
    try:
        module = importlib.import_module(module_path_str)
        stats_func = getattr(module, expected_func_name)
    except ImportError:
        print(f"Error: could not import module '{module_path_str}'.")
        raise
    except AttributeError:
        print(f"Error: could not find '{expected_func_name}' in '{module_path_str}'.")
        raise
    print(f"Loaded stats function '{expected_func_name}' from '{module_path_str}'")
    return stats_func()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate two D&D combat policies head-to-head (symmetric).")
    parser.add_argument("--agent1_stats_module", type=str, required=True,
                        help="Module path for Agent 1's stats (e.g. 'bestiary.wolf').")
    parser.add_argument("--agent2_stats_module", type=str, required=True,
                        help="Module path for Agent 2's stats (e.g. 'bestiary.goblin').")
    parser.add_argument("--agent1_model", type=str, default=None,
                        help="Path to Agent 1's model. Omit to have Agent 1 act randomly.")
    parser.add_argument("--agent2_model", type=str, default=None,
                        help="Path to Agent 2's model. Omit to have Agent 2 act randomly.")
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="Number of episodes to run. Default: 1000")
    parser.add_argument("--map_width", type=int, default=10, help="Map width.")
    parser.add_argument("--map_height", type=int, default=10, help="Map height.")
    parser.add_argument("--max_rounds", type=int, default=60,
                        help="Max rounds (Agent 1 turn + Agent 2 turn) before a draw. Default: 60")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; episode N uses seed+N so runs are reproducible.")
    parser.add_argument("--swap_sides", action="store_true",
                        help="Also run the mirror matchup (each model in the other seat) and "
                             "report both, cancelling out any first-turn advantage.")
    return parser.parse_args()


def run_matchup(agent1_stats, agent2_stats, agent1_model, agent2_model,
                args, header: str) -> dict:
    """Run num_episodes of one matchup. Returns win/loss/draw tallies."""
    print(f"\n=== {header} ===")
    env = DnDCombatEnv(
        map_width=args.map_width,
        map_height=args.map_height,
        agent_stats=agent1_stats,
        enemy_stats=agent2_stats,
        grid_size=1,
        render_mode=None,
        export_frames_path=None,
        enemy_model_path=None,  # both sides are driven here, not internally
    )
    flatten = FlattenObservation(env)

    policies = {
        'agent': spectator.load_policy(agent1_model, "Agent 1"),
        'enemy': spectator.load_policy(agent2_model, "Agent 2"),
    }
    rng = np.random.default_rng(args.seed)

    tally = {'agent1': 0, 'agent2': 0, 'draw': 0}
    start_time = time.time()
    print(f"Running {args.num_episodes} episodes...")

    for episode_num in range(1, args.num_episodes + 1):
        outcome = spectator.play_episode(
            env, flatten, policies, rng,
            max_rounds=args.max_rounds,
            seed=args.seed + episode_num - 1,
        )
        tally[outcome] += 1

        if episode_num % 100 == 0 or episode_num == args.num_episodes:
            elapsed = time.time() - start_time
            print(f"  {episode_num}/{args.num_episodes} | "
                  f"Agent 1: {tally['agent1']}  Agent 2: {tally['agent2']}  "
                  f"Draws: {tally['draw']} | {elapsed:.1f}s")

    env.close()
    return tally


def report(tally: dict, total: int, label1: str, label2: str) -> None:
    def pct(n: int) -> str:
        return f"{n} ({n / total * 100:.1f}%)" if total else "0"
    print(f"  {label1:<28} {pct(tally['agent1'])}")
    print(f"  {label2:<28} {pct(tally['agent2'])}")
    print(f"  {'Draws':<28} {pct(tally['draw'])}")


def main():
    args = parse_arguments()
    if not SB3_AVAILABLE:
        print("Gymnasium is required to run this evaluation script.")
        return

    print("D&D Head-to-Head Evaluation")
    print("------------------------------------")
    print(f"Agent 1: {args.agent1_stats_module} | model: {args.agent1_model or 'random'}")
    print(f"Agent 2: {args.agent2_stats_module} | model: {args.agent2_model or 'random'}")
    print(f"Episodes: {args.num_episodes} | Map: {args.map_width}x{args.map_height} | "
          f"Max rounds: {args.max_rounds} | Seed: {args.seed}")
    print("------------------------------------")

    print("\nLoading creature stats...")
    try:
        agent1_stats = load_stats_from_module(args.agent1_stats_module)
        agent2_stats = load_stats_from_module(args.agent2_stats_module)
    except Exception:
        print("Failed to load creature stats. Exiting.")
        return

    creature1 = args.agent1_stats_module.split('.')[-1]
    creature2 = args.agent2_stats_module.split('.')[-1]

    tally = run_matchup(
        agent1_stats, agent2_stats, args.agent1_model, args.agent2_model, args,
        f"Agent 1 ({creature1}) vs Agent 2 ({creature2})")

    print("\n------------------------------------")
    print("           Results")
    print("------------------------------------")
    report(tally, args.num_episodes,
           f"Agent 1 ({creature1}) wins", f"Agent 2 ({creature2}) wins")

    if args.swap_sides:
        # Mirror matchup: the same two policies with seats exchanged. Agent 1
        # always acts first, so comparing both directions separates real skill
        # from the first-turn advantage.
        swapped = run_matchup(
            agent2_stats, agent1_stats, args.agent2_model, args.agent1_model, args,
            f"Swapped: Agent 1 ({creature2}) vs Agent 2 ({creature1})")
        print("\n------------------------------------")
        print("      Results (sides swapped)")
        print("------------------------------------")
        report(swapped, args.num_episodes,
               f"Agent 1 ({creature2}) wins", f"Agent 2 ({creature1}) wins")

        total = args.num_episodes * 2
        m1 = tally['agent1'] + swapped['agent2']
        m2 = tally['agent2'] + swapped['agent1']
        draws = tally['draw'] + swapped['draw']
        print("\n------------------------------------")
        print("   Combined (seat advantage removed)")
        print("------------------------------------")
        print(f"  {creature1 + ' policy wins':<28} {m1} ({m1 / total * 100:.1f}%)")
        print(f"  {creature2 + ' policy wins':<28} {m2} ({m2 / total * 100:.1f}%)")
        print(f"  {'Draws':<28} {draws} ({draws / total * 100:.1f}%)")

    print("------------------------------------")


if __name__ == "__main__":
    main()
