"""Action-mask validity and mask/executor consistency."""
import os

import numpy as np
import pytest
from gymnasium import spaces

from sunztu.creature import Creature
from sunztu.dice import roll
from sunztu.env import DnDCombatEnv
from bestiary.bandit import get_bandit_stats
from bestiary.cat import get_cat_stats
from bestiary.commoner import get_commoner_stats
from bestiary.goblin import get_goblin_stats
from bestiary.goblin import get_goblin_stats as _goblin_stats
from bestiary.guard import get_guard_stats
from bestiary.guard import get_guard_stats as _guard_stats
from bestiary.scout import get_scout_stats
from bestiary.scout import get_scout_stats as _scout_stats
from bestiary.wolf import get_wolf_stats
from bestiary.wolf import get_wolf_stats as _wolf_stats
from conftest import get_action_idx, make_roller, find_action, WATCHER, EXPECTED_STAT_KEYS


def test_action_masks_reflect_state(dnd_env):
    dnd_env.reset(seed=500)
    agent = dnd_env.agent
    agent.position = [5, 5]
    dnd_env.enemy.position = [5, 6]  # adjacent

    def mask_for(name):
        return bool(dnd_env.action_masks()[find_action(dnd_env, name)])

    # Fresh turn, adjacent enemy
    assert mask_for("attack_longsword")
    assert mask_for("dash") and mask_for("dodge") and mask_for("disengage")
    assert mask_for("grapple") and mask_for("shove_prone") and mask_for("shove_push")
    assert not mask_for("hide")            # adjacent
    assert not mask_for("search")          # nothing hidden
    assert not mask_for("stand_up")        # not prone
    assert not mask_for("escape_grapple")  # not grappled
    assert not mask_for("move_S")          # enemy occupies [5,6]
    assert mask_for("move_N")
    assert mask_for("end_turn")

    # After the Action is spent (no Multiattack slots left)
    agent.used_action_this_turn = True
    agent.attacks_remaining = 0
    assert not mask_for("attack_longsword")
    assert not mask_for("dash")
    assert not mask_for("grapple")
    assert mask_for("end_turn")

    # Grappled: no moves, escape available with a fresh Action
    agent.used_action_this_turn = False
    agent.conditions.add("grappled")
    assert not mask_for("move_N")
    assert mask_for("escape_grapple")
    agent.conditions.discard("grappled")

    # Prone with 1 movement left: cannot stand (costs 3) or crawl (costs 2)
    agent.conditions.add("prone")
    agent.speed_remaining = 1
    assert not mask_for("stand_up")
    assert not mask_for("move_N")
    agent.speed_remaining = 6
    assert mask_for("stand_up")

    # Hidden enemy makes search valid; distance makes hide valid
    agent.conditions.discard("prone")
    dnd_env.enemy.conditions.add("hidden")
    dnd_env.enemy.position = [9, 9]
    assert mask_for("search")
    assert mask_for("hide")

def test_masked_valid_actions_always_execute(dnd_env):
    """Property check: any action the mask marks valid must be accepted by the
    executor (mask and executor validity rules must never drift apart)."""
    import random as _random
    _random.seed(99)
    dnd_env.reset(seed=501)
    end_turn_idx = find_action(dnd_env, "end_turn")

    for i in range(300):
        mask = dnd_env.action_masks()
        valid_ids = [j for j, ok in enumerate(mask) if ok]
        assert valid_ids, "mask must never be all-False (end_turn is always legal)"
        action = _random.choice(valid_ids)
        decoded = dnd_env._decode_action(action)

        check_info = {}
        _, terminated, success = dnd_env._execute_micro_action(
            dnd_env.agent, dnd_env.enemy, decoded, check_info, 'agent')
        assert success or terminated, (
            f"masked-valid action {decoded['name']} was refused: {check_info}")

        if terminated or not dnd_env.enemy.is_alive:
            dnd_env.reset(seed=502 + i)
        elif action == end_turn_idx or dnd_env._turn_economy_exhausted(dnd_env.agent):
            # Simulate the turn handover so state keeps evolving
            dnd_env.agent.start_new_turn()
