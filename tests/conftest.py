"""Shared fixtures and helpers for the sunztu test suite."""
import os
from typing import Optional

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


EXPECTED_STAT_KEYS = ["ac", "max_hp", "speed_total", "attacks", "bonus_actions",
                      "abilities", "skills", "passive_perception", "traits"]

WATCHER = {"ac": 10, "max_hp": 20, "speed_total": 6, "attacks": [], "bonus_actions": []}

@pytest.fixture
def sample_creature_stats():
    return {
        "name": "Test Dummy",
        "ac": 10,
        "max_hp": 20,
        "speed_total": 6, # cells
        "attacks": [
            {"name": "punch", "to_hit": 3, "damage_dice": "1d4+1", "num_attacks": 1},
            {"name": "kick", "to_hit": 3, "damage_dice": "1d6+1", "num_attacks": 1}
        ],
        "bonus_actions": ["minor_heal_self_1hp"],
        "initial_position": [0, 0]
    }

@pytest.fixture
def agent_stats_env():
    return {
        "ac": 15, "max_hp": 50, "speed_total": 6,
        "attacks": [{"name": "longsword", "to_hit": 5, "damage_dice": "1d8+3", "num_attacks": 1}],
        "bonus_actions": ["bonus_move_1_cell"]
    }

@pytest.fixture
def enemy_stats_env():
    return {
        "ac": 13, "max_hp": 30, "speed_total": 5,
        "attacks": [{"name": "scimitar", "to_hit": 4, "damage_dice": "1d6+2", "num_attacks": 1}],
        "bonus_actions": []
    }

@pytest.fixture
def dnd_env(agent_stats_env, enemy_stats_env):
    # Default to ansi mode for most tests to avoid Pygame initialization issues in CI
    return DnDCombatEnv(map_width=10, map_height=10, 
                        agent_stats=agent_stats_env, 
                        enemy_stats=enemy_stats_env,
                        render_mode='ansi')

@pytest.fixture
def scout_vs_wall_env():
    """Scout agent vs an unhittable, harmless enemy for multiattack tests."""
    tough_enemy = {
        "ac": 100, "max_hp": 50, "speed_total": 6,
        "attacks": [], "bonus_actions": []
    }
    env = DnDCombatEnv(map_width=10, map_height=10,
                       agent_stats=_scout_stats(), enemy_stats=tough_enemy,
                       render_mode=None)
    env.reset(seed=300)
    env.agent.position = [0, 0]
    env.enemy.position = [0, 1]  # adjacent
    return env

def get_action_idx(env_instance, action_type_or_name: str, attack_name: Optional[str] = None) -> int:
    """Helper to find the index of an action in the action_map."""
    for i, action_def in enumerate(env_instance.action_map):
        if action_def["type"] == action_type_or_name:
            if action_type_or_name == "attack" and attack_name:
                # Ensure attack_name is from the creature's attack list for safety
                # This part of the logic might need to be more robust if names can clash
                # or if index is preferred. For now, matching name as per current structure.
                if hasattr(env_instance, 'agent') and action_def['name'] == f"attack_{attack_name}":
                    return i
                elif hasattr(env_instance, 'enemy') and action_def['name'] == f"attack_{attack_name}": # Check enemy too if context implies
                    return i
            elif action_type_or_name != "attack": # For non-attack types, type matching is enough, or use name
                 if action_def["name"] == action_type_or_name: # e.g. "dash", "dodge"
                    return i
        # Allow searching by just the unique name if it's not a generic type like "attack"
        if action_def["name"] == action_type_or_name and \
           action_type_or_name not in ["attack", "move", "bonus_action"]: # Avoid overly generic name matches for these types
             return i


    # Fallback for simple types if name wasn't specific enough initially
    # and no specific sub_name (like attack_name) was provided.
    if action_type_or_name in ["attack", "move", "bonus_action", "end_turn", "dash", "disengage", "dodge"] and not attack_name:
        for i, action_def in enumerate(env_instance.action_map):
             if action_def["type"] == action_type_or_name:
                  return i # Return first match for generic types if no sub-name given

    raise ValueError(f"Action '{action_type_or_name}' (Attack Name: '{attack_name}') not found in env action_map: {env_instance.action_map}")

def make_roller(values):
    """Deterministic dice roller that returns the given values in order."""
    seq = list(values)
    def roller(dice_string):
        return seq.pop(0)
    return roller

def find_action(env, name):
    for i, a in enumerate(env.action_map):
        if a["name"] == name:
            return i
    raise ValueError(f"action {name} not in map")
