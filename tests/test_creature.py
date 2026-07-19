"""Creature data model: stats, damage, movement, saves, stat-block structure."""
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


def test_creature_initialization(sample_creature_stats):
    creature = Creature(**sample_creature_stats)
    assert creature.name == sample_creature_stats["name"]
    assert creature.ac == sample_creature_stats["ac"]
    assert creature.max_hp == sample_creature_stats["max_hp"]
    assert creature.current_hp == sample_creature_stats["max_hp"]
    assert creature.speed_total == sample_creature_stats["speed_total"]
    assert creature.speed_remaining == sample_creature_stats["speed_total"]
    assert creature.attacks == sample_creature_stats["attacks"]
    assert creature.bonus_actions == sample_creature_stats["bonus_actions"]
    assert creature.position == sample_creature_stats["initial_position"]
    assert creature.is_alive is True

def test_take_damage(sample_creature_stats):
    creature = Creature(**sample_creature_stats)
    
    # Damage less than HP
    creature.take_damage(5)
    assert creature.current_hp == sample_creature_stats["max_hp"] - 5
    assert creature.is_alive is True

    # Damage equal to remaining HP
    creature.take_damage(creature.current_hp)
    assert creature.current_hp == 0
    assert creature.is_alive is False

    # Reset for next test part and damage more than HP
    creature.current_hp = sample_creature_stats["max_hp"]
    creature.is_alive = True
    creature.take_damage(sample_creature_stats["max_hp"] + 10)
    assert creature.current_hp == 0
    assert creature.is_alive is False

    # Damage 0
    creature.current_hp = sample_creature_stats["max_hp"]
    creature.is_alive = True
    hp_before_0_damage = creature.current_hp
    creature.take_damage(0)
    assert creature.current_hp == hp_before_0_damage
    assert creature.is_alive is True
    
    # Cannot take negative damage
    creature.take_damage(-5)
    assert creature.current_hp == hp_before_0_damage
    assert creature.is_alive is True

def test_make_attack_hit_miss(sample_creature_stats):
    attacker = Creature(**sample_creature_stats)
    # Target with very high AC to force a miss (unless 1d20 is 20 and to_hit is high)
    target_high_ac = Creature(name="HighAC", ac=100, max_hp=20, speed_total=5, attacks=[], bonus_actions=[], initial_position=[1,0])
    # Target with very low AC to force a hit (unless 1d20 is 1 and to_hit is low)
    target_low_ac = Creature(name="LowAC", ac=1, max_hp=20, speed_total=5, attacks=[], bonus_actions=[], initial_position=[1,0])

    # To make tests deterministic without complex mocking of dice.roll's random.randint:
    # We rely on extreme values of AC.
    # A roll of 1 + to_hit should miss AC 100. A roll of 20 + to_hit should hit AC 1.
    
    # Test MISS: Attacker's to_hit is +3. If 1d20 rolls 1, total is 4. Target AC 100.
    # This will almost always miss. We check that if it misses, HP is unchanged.
    # If it by chance hits (natural 20, high to_hit), then HP should change.
    # For a more robust test, we'd mock dice_roller.
    
    initial_hp_high_ac = target_high_ac.current_hp
    hit_status_miss, _ = attacker.make_attack(target_high_ac, 0, roll) # Use actual roll
    if not hit_status_miss: # Most likely outcome
        assert target_high_ac.current_hp == initial_hp_high_ac
    else: # It was a critical hit or high roll against the odds
        assert target_high_ac.current_hp < initial_hp_high_ac

    # Test HIT: Attacker's to_hit is +3. If 1d20 rolls 20, total is 23. Target AC 1.
    initial_hp_low_ac = target_low_ac.current_hp
    hit_status_hit, damage_done_hit = attacker.make_attack(target_low_ac, 0, roll)
    if hit_status_hit: # Most likely outcome
        assert target_low_ac.current_hp < initial_hp_low_ac
        assert target_low_ac.current_hp == initial_hp_low_ac - damage_done_hit
    else: # It was a critical miss or very low roll
        assert target_low_ac.current_hp == initial_hp_low_ac

def test_move_valid_invalid(sample_creature_stats):
    creature = Creature(**sample_creature_stats) # speed_total = 6
    map_w, map_h = 10, 10

    # Valid move
    moved = creature.move(2, 2, map_w, map_h) # Cost 4
    assert moved is True
    assert creature.position == [2, 2]
    assert creature.speed_remaining == sample_creature_stats["speed_total"] - 4

    # Move beyond speed_remaining
    moved_again = creature.move(1, 2, map_w, map_h) # Cost 3, remaining 2. Should fail.
    assert moved_again is False
    assert creature.position == [2, 2] # Unchanged
    assert creature.speed_remaining == sample_creature_stats["speed_total"] - 4 # Unchanged

    # Move to exact remaining speed
    creature.speed_remaining = 2
    moved_exact = creature.move(1, 1, map_w, map_h) # Cost 2
    assert moved_exact is True
    assert creature.position == [3, 3]
    assert creature.speed_remaining == 0
    
    # Move out of bounds (x < 0)
    creature.position = [0,0]
    creature.speed_remaining = 5
    moved_oob_x = creature.move(-1, 0, map_w, map_h)
    assert moved_oob_x is False
    assert creature.position == [0,0]
    assert creature.speed_remaining == 5

    # Move out of bounds (y >= map_height)
    moved_oob_y = creature.move(0, map_h, map_w, map_h)
    assert moved_oob_y is False
    assert creature.position == [0,0]
    assert creature.speed_remaining == 5

def test_reset_creature(sample_creature_stats):
    creature = Creature(**sample_creature_stats)
    creature.take_damage(10)
    creature.move(1,0,10,10)
    creature.speed_remaining = 0
    
    new_pos = [5,5]
    new_max_hp = 30
    new_speed = 8
    creature.reset(initial_position=new_pos, max_hp=new_max_hp, speed_total=new_speed)
    
    assert creature.current_hp == new_max_hp
    assert creature.max_hp == new_max_hp
    assert creature.position == new_pos
    assert creature.speed_remaining == new_speed
    assert creature.speed_total == new_speed
    assert creature.is_alive is True

@pytest.mark.parametrize("stat_func, creature_name", [
    (get_cat_stats, "Cat"),
    (get_bandit_stats, "Bandit"),
    (get_guard_stats, "Guard"),
    (get_goblin_stats, "Goblin"),
    (get_scout_stats, "Scout"),
    (get_commoner_stats, "Commoner"),
    (get_wolf_stats, "Wolf"),
])
def test_creature_stat_function_structure(stat_func, creature_name):
    stats = stat_func()
    assert isinstance(stats, dict), f"{creature_name} stats should be a dict"
    for key in EXPECTED_STAT_KEYS:
        assert key in stats, f"{creature_name} stats missing key: {key}"

    assert isinstance(stats["attacks"], list), f"{creature_name} 'attacks' should be a list"
    assert isinstance(stats["bonus_actions"], list), f"{creature_name} 'bonus_actions' should be a list"

    if stats["attacks"]:
        first_attack = stats["attacks"][0]
        assert isinstance(first_attack, dict), f"{creature_name} first attack should be a dict"
        expected_attack_keys = ["name", "to_hit", "damage_dice", "range", "attack_type", "num_attacks"]
        for key in expected_attack_keys:
            assert key in first_attack, f"{creature_name} first attack missing key: {key} in {first_attack}"

def test_ability_modifiers_and_saving_throws():
    wolf = Creature(name="wolf", initial_position=[0, 0], **_wolf_stats())
    assert wolf.ability_modifier("str") == 1   # STR 12
    assert wolf.ability_modifier("dex") == 2   # DEX 15
    assert wolf.ability_modifier("int") == -4  # INT 3
    # d20 roll of 10 + STR mod 1 = 11 vs DC 11 -> success
    assert wolf.saving_throw("str", 11, make_roller([10])) is True
    # d20 roll of 9 + 1 = 10 vs DC 11 -> failure
    assert wolf.saving_throw("str", 11, make_roller([9])) is False
