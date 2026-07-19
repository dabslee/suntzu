"""Combat mechanics: action economy, multiattack, conditions, stealth, contests."""
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


def test_action_economy_rules(dnd_env):
    dnd_env.reset(seed=100)
    agent = dnd_env.agent

    attack_idx = get_action_idx(dnd_env, "attack", agent.attacks[0]["name"])
    dash_idx = get_action_idx(dnd_env, "dash")
    bonus_action_idx = get_action_idx(dnd_env, dnd_env.agent.bonus_actions[0]) if dnd_env.agent.bonus_actions else -1
    move_e_idx = get_action_idx(dnd_env, "move_E")

    # 1. Agent attempts two main actions: Attack then Dash. Dash should not effectively happen if action is used.
    agent.start_new_turn() # Ensure fresh turn
    obs, reward, term, trunc, info = dnd_env.step(attack_idx)
    assert agent.used_action_this_turn is True
    # Store speed after attack, before dash attempt
    speed_after_attack = agent.speed_remaining

    obs, reward, term, trunc, info = dnd_env.step(dash_idx) # Attempt Dash
    assert info.get('agent_action_details') == "Agent tried Dash but action already used." # Check info log
    assert agent.took_dash_action_this_turn is False # Dash effect should not apply
    assert agent.speed_remaining == speed_after_attack # Speed should not have doubled

    # 2. Agent attempts two bonus actions. (Assuming agent has at least one bonus action)
    if bonus_action_idx != -1:
        agent.start_new_turn()
        obs, reward, term, trunc, info = dnd_env.step(bonus_action_idx)
        assert agent.used_bonus_action_this_turn is True

        # Try another bonus action (or the same one) - should fail
        obs, reward, term, trunc, info = dnd_env.step(bonus_action_idx)
        assert "Bonus action already used" in info.get('agent_action_details', "") or \
               "Agent tried Bonus Action but it was already used" in info.get('agent_action_details', "")
    else:
        pytest.skip("Agent has no bonus actions to test double usage.")

    # 3. Agent uses action, bonus action, and moves. All should be successful.
    agent.start_new_turn()
    initial_speed = agent.speed_remaining

    # Action
    obs, reward, term, trunc, info = dnd_env.step(attack_idx)
    assert agent.used_action_this_turn is True
    assert not term # Assuming attack doesn't end episode for this test

    # Bonus Action
    if bonus_action_idx != -1:
        obs, reward, term, trunc, info = dnd_env.step(bonus_action_idx)
        assert agent.used_bonus_action_this_turn is True
        assert not term

    # Move
    obs, reward, term, trunc, info = dnd_env.step(move_e_idx)
    assert agent.speed_remaining < initial_speed # Speed should be consumed by move
    assert not term

def test_dash_action(dnd_env):
    dnd_env.reset(seed=101)
    agent = dnd_env.agent
    initial_total_speed = agent.speed_total

    dash_idx = get_action_idx(dnd_env, "dash")
    move_e_idx = get_action_idx(dnd_env, "move_E")

    # Agent takes Dash action
    agent.start_new_turn()
    obs, reward, term, trunc, info = dnd_env.step(dash_idx)

    assert agent.used_action_this_turn is True
    assert agent.took_dash_action_this_turn is True
    # Speed remaining should be initial_total_speed (from start_new_turn) + initial_total_speed (from dash)
    assert agent.speed_remaining == initial_total_speed * 2

    # Agent moves using dashed speed
    obs, reward, term, trunc, info = dnd_env.step(move_e_idx) # Move 1 cell
    assert agent.speed_remaining == (initial_total_speed * 2) - 1

    # Simulate passing turn / enemy turn, then start agent's new turn
    # This typically happens in the main loop or by enemy action.
    # For this test, manually call start_new_turn for the agent.
    agent.start_new_turn() # This should reset speed and dash status
    assert agent.took_dash_action_this_turn is False
    assert agent.speed_remaining == initial_total_speed

def test_disengage_and_opportunity_attack(dnd_env):
    dnd_env.reset(seed=102)
    agent = dnd_env.agent
    enemy = dnd_env.enemy

    disengage_idx = get_action_idx(dnd_env, "disengage")
    move_e_idx = get_action_idx(dnd_env, "move_E") # Action to move away
    pass_idx = get_action_idx(dnd_env, "end_turn")

    # Scenario 1: OA Triggered
    agent.start_new_turn()
    enemy.start_new_turn() # Ensure enemy reaction is available
    agent.position = [1,1]
    enemy.position = [1,2] # Adjacent
    enemy.current_hp = 100 # Ensure enemy survives to make OA
    initial_agent_hp = agent.current_hp

    # Agent moves away without Disengaging
    obs, reward_oa, term_oa, trunc_oa, info_oa = dnd_env.step(move_e_idx)

    assert enemy.used_reaction_this_round is True, "Enemy should have used reaction for OA"
    # Check if agent took damage (this assumes OA hits, might need dice mocking for perfect test)
    # For now, if reaction used, assume OA was processed. Damage depends on hit.
    if 'opportunity_attack_by_enemy_outcome' in info_oa and "Hit" in info_oa['opportunity_attack_by_enemy_outcome']:
        assert agent.current_hp < initial_agent_hp, "Agent should take damage from OA"

    # Scenario 2: Disengage Prevents OA
    dnd_env.reset(seed=103) # Reset positions and states
    agent.start_new_turn()
    enemy.start_new_turn()
    agent.position = [1,1]
    enemy.position = [1,2] # Adjacent
    enemy.used_reaction_this_round = False # Ensure reaction available
    initial_agent_hp_disengage = agent.current_hp

    # Agent takes Disengage action
    obs, _, _, _, _ = dnd_env.step(disengage_idx)
    assert agent.is_disengaging is True
    assert agent.used_action_this_turn is True

    # Agent moves away from adjacent enemy
    obs, _, _, _, info_no_oa = dnd_env.step(move_e_idx)
    assert enemy.used_reaction_this_round is False, "Enemy should not use reaction if agent disengaged"
    assert agent.current_hp == initial_agent_hp_disengage, "Agent should not take damage if enemy OA prevented"
    assert 'opportunity_attack_by_enemy' not in info_no_oa # Check that no OA was logged

    # Scenario 3: No Reaction for OA
    dnd_env.reset(seed=104)
    agent.start_new_turn()
    enemy.start_new_turn()
    agent.position = [1,1]
    enemy.position = [1,2] # Adjacent
    enemy.used_reaction_this_round = True # Enemy already used reaction
    initial_agent_hp_no_reaction = agent.current_hp

    # Agent moves away without Disengaging
    obs, _, _, _, info_no_reaction_avail = dnd_env.step(move_e_idx)
    assert enemy.used_reaction_this_round is True # Still true, but no new OA
    assert agent.current_hp == initial_agent_hp_no_reaction, "Agent should not take damage if enemy has no reaction"
    assert 'opportunity_attack_by_enemy' not in info_no_reaction_avail

def test_dodge_action(dnd_env):
    dnd_env.reset(seed=105)
    agent = dnd_env.agent
    enemy = dnd_env.enemy

    dodge_idx = get_action_idx(dnd_env, "dodge")
    pass_idx = get_action_idx(dnd_env, "end_turn") # Agent passes so enemy can attack

    # Agent takes Dodge action
    agent.start_new_turn()
    obs, _, _, _, _ = dnd_env.step(dodge_idx)
    assert agent.is_dodging is True
    assert agent.used_action_this_turn is True

    # Enemy attacks agent. For this test, we assume the enemy AI will attack.
    # Place them adjacent.
    agent.position = [1,1]
    enemy.position = [1,2]
    enemy.start_new_turn() # Enemy's turn

    # Agent needs to take a non-action (like pass) for the step to proceed to enemy turn
    obs, _, _, _, info_enemy_attack = dnd_env.step(pass_idx)

    # Verify the enemy's attack was made at disadvantage (intent)
    # This relies on the step function correctly identifying agent.is_dodging
    # and passing 'disadvantage' to the make_attack call for the enemy.
    # We check the info dict for evidence of this.
    assert 'enemy_attack_modifier' in info_enemy_attack, "Enemy attack modifier info missing"
    assert "Agent dodging, attack at disadvantage" in info_enemy_attack['enemy_attack_modifier'], \
        "Enemy attack should be at disadvantage due to agent dodging"

    # On agent's next turn, is_dodging should be false before it acts.
    agent.start_new_turn() # This is called at the start of agent's turn processing in step()
    assert agent.is_dodging is False # start_new_turn resets it

def test_ranged_attack_rules(dnd_env, agent_stats_env, enemy_stats_env):
    # Modify agent to have a ranged attack for this test
    # This requires a new env instance with custom agent stats for this test
    ranged_attack_stats = agent_stats_env.copy()
    ranged_attack_stats["attacks"] = [
        {"name": "shortbow", "to_hit": 5, "damage_dice": "1d6+3", "num_attacks": 1, "attack_type": "ranged", "range": 16} # 80ft/5ft
    ]
    # Create a new environment instance with this specific agent
    test_env = DnDCombatEnv(map_width=10, map_height=10,
                            agent_stats=ranged_attack_stats,
                            enemy_stats=enemy_stats_env,
                            render_mode='ansi')
    test_env.reset(seed=106)
    agent = test_env.agent
    enemy = test_env.enemy

    ranged_attack_idx = get_action_idx(test_env, "attack", "shortbow")

    # Scenario 1: Ranged attack with adjacent enemy
    agent.start_new_turn()
    agent.position = [1,1]
    enemy.position = [1,2] # Adjacent

    obs, _, _, _, info_adj = test_env.step(ranged_attack_idx)
    assert 'agent_attack_modifier' in info_adj, "Agent attack modifier info missing for adjacent ranged"
    assert "Ranged attack at disadvantage (enemy adjacent)" in info_adj['agent_attack_modifier'], \
        "Agent's ranged attack should be at disadvantage when enemy is adjacent"

    # Scenario 2: Ranged attack with no adjacent enemy
    test_env.reset(seed=107) # Reset positions
    agent.start_new_turn()
    agent.position = [1,1]
    enemy.position = [5,5] # Not adjacent

    obs, _, _, _, info_not_adj = test_env.step(ranged_attack_idx)
    assert 'agent_attack_modifier' not in info_not_adj or \
           "Ranged attack at disadvantage (enemy adjacent)" not in info_not_adj.get('agent_attack_modifier',''), \
           "Agent's ranged attack should not be at disadvantage from proximity if enemy not adjacent"

def test_new_observation_fields(dnd_env):
    dnd_env.reset(seed=108)
    agent = dnd_env.agent
    enemy = dnd_env.enemy # For OA

    obs, _ = dnd_env.reset() # Initial state
    assert obs["agent_used_action"][0] == 0.0
    assert obs["agent_used_bonus_action"][0] == 0.0
    assert obs["agent_used_reaction"][0] == 0.0
    assert obs["agent_is_dodging"][0] == 0.0

    # Test after Dodge
    dodge_idx = get_action_idx(dnd_env, "dodge")
    obs, _, _, _, _ = dnd_env.step(dodge_idx)
    assert obs["agent_is_dodging"][0] == 1.0
    assert obs["agent_used_action"][0] == 1.0 # Dodge uses an action

    # Reset for next part of test (new turn implicitly handled by step if first action)
    # To be sure, manually start turn or take action that implies new turn
    dnd_env.reset(seed=109) # Full reset for clean state
    obs, _ = dnd_env.reset()

    # Test after main action (e.g. Attack)
    attack_idx = get_action_idx(dnd_env, "attack", agent.attacks[0]["name"])
    obs, _, _, _, _ = dnd_env.step(attack_idx)
    assert obs["agent_used_action"][0] == 1.0

    # Test after bonus action (if available)
    if agent.bonus_actions:
        dnd_env.reset(seed=110)
        obs, _ = dnd_env.reset()
        bonus_action_idx = get_action_idx(dnd_env, agent.bonus_actions[0])
        obs, _, _, _, _ = dnd_env.step(bonus_action_idx)
        assert obs["agent_used_bonus_action"][0] == 1.0

    # Test after reaction (via enemy OA on agent)
    dnd_env.reset(seed=111)
    agent.start_new_turn() # Agent's turn
    enemy.start_new_turn() # Enemy's turn (so reaction is available)
    agent.position = [1,1]
    enemy.position = [1,2] # Adjacent
    # Agent moves away to trigger OA from enemy
    move_e_idx = get_action_idx(dnd_env, "move_E")
    obs, _, _, _, _ = dnd_env.step(move_e_idx) # This obs is for the agent AFTER its move and enemy's OA
                                            # The enemy's reaction status is on enemy, not agent obs
                                            # So this observation won't reflect enemy's reaction.

    # To test agent_used_reaction, the *agent* must make an OA.
    # So, enemy moves, agent makes OA.
    dnd_env.reset(seed=112)
    agent.start_new_turn()
    enemy.start_new_turn()
    agent.position = [1,2] # Agent
    enemy.position = [1,1] # Enemy adjacent

    # Enemy needs to move away from agent. We need to make enemy move.
    # Easiest way: agent passes, enemy AI (rule-based) tries to move if it can't attack.
    # Give enemy no attacks so it's forced to consider moving.
    original_enemy_attacks = enemy.attacks
    enemy.attacks = []
    pass_idx = get_action_idx(dnd_env, "end_turn")
    obs, _, _, _, info = dnd_env.step(pass_idx) # Agent passes, enemy turn happens.

    # Now check agent's observation from THIS step (which includes enemy turn)
    # The observation 'obs' is from the perspective of the *next* agent action.
    # If agent made an OA, its used_reaction should be 1.0.
    # The info dict from the step where enemy moved and agent OAd will show agent OA.
    if 'opportunity_attack_by_agent' in info:
        # The 'obs' is for the *start* of the agent's next turn.
        # Reaction resets at start of turn. So this test for obs["agent_used_reaction"]
        # needs to be more nuanced or check creature state directly before obs is made.
        # For now, we check the flag on the creature after the step.
        assert dnd_env.agent.used_reaction_this_round is True

        # If we get a new obs *after* this step, for the *next* agent decision:
        # The used_reaction_this_round would have been reset by agent.start_new_turn().
        # So, let's check the obs provided *by the step that included the reaction*.
        # The 'obs' returned by dnd_env.step(pass_idx) is the one for the agent's *next* turn.
        # We need to check the state of agent.used_reaction_this_round *before* _get_obs() is called for that next turn.

        # This part of the test highlights that observation reflects state *for the next decision*.
        # The fact an OA happened and reaction was used is better tested by creature state or logs.
        # The obs["agent_used_reaction"] will be 0.0 if it's a new turn for the agent.
        # However, if the test framework allows, one could inspect the observation
        # for the *enemy* if it's model-based, as that would be generated mid-step.

    enemy.attacks = original_enemy_attacks # Restore

def test_advantage_disadvantage_stacking(dnd_env):
    dnd_env.reset(seed=200)
    agent, enemy = dnd_env.agent, dnd_env.enemy
    attack = agent.attacks[0]  # melee longsword

    # No modifiers -> straight roll
    status, msgs = dnd_env._resolve_attack_modifiers(agent, enemy, attack, 1, "agent", "enemy")
    assert status is None and msgs == []

    # Target prone and adjacent -> advantage
    enemy.conditions.add("prone")
    status, _ = dnd_env._resolve_attack_modifiers(agent, enemy, attack, 1, "agent", "enemy")
    assert status == 'advantage'

    # Add a disadvantage source (attacker also prone): adv + dis -> straight roll
    agent.conditions.add("prone")
    status, msgs = dnd_env._resolve_attack_modifiers(agent, enemy, attack, 1, "agent", "enemy")
    assert status is None
    assert any("cancel" in m for m in msgs)

    # Two disadvantage sources and one advantage source still cancel to straight
    enemy.is_dodging = True
    status, _ = dnd_env._resolve_attack_modifiers(agent, enemy, attack, 1, "agent", "enemy")
    assert status is None

    # Prone target attacked from range -> disadvantage (not advantage)
    agent.conditions.discard("prone")
    enemy.is_dodging = False
    status, _ = dnd_env._resolve_attack_modifiers(agent, enemy, attack, 3, "agent", "enemy")
    assert status == 'disadvantage'

def test_multiattack_grants_two_attacks(scout_vs_wall_env):
    env = scout_vs_wall_env
    shortsword_idx = find_action(env, "attack_shortsword")

    obs, _, _, _, info = env.step(shortsword_idx)
    assert info["agent_attack_outcome"] == "Miss."  # AC 100 is unhittable
    assert env.agent.attacks_remaining == 1
    assert obs["agent_attacks_remaining_norm"][0] == pytest.approx(0.5)

    obs, _, _, _, info = env.step(shortsword_idx)
    assert info["agent_attack_outcome"] == "Miss."
    assert env.agent.attacks_remaining == 0

    # Third attack: the Attack action is spent and no slots remain
    obs, _, _, _, info = env.step(shortsword_idx)
    assert info["agent_attack_outcome"] == "Action already used."

def test_multiattack_type_lock(scout_vs_wall_env):
    env = scout_vs_wall_env
    shortsword_idx = find_action(env, "attack_shortsword")
    longbow_idx = find_action(env, "attack_longbow")

    env.step(shortsword_idx)  # first attack locks Multiattack to melee
    obs, _, _, _, info = env.step(longbow_idx)
    assert info["agent_attack_outcome"] == "Wrong attack type for Multiattack."
    assert env.agent.attacks_remaining == 1  # slot not consumed by the refusal

    # The locked-in type still works
    obs, _, _, _, info = env.step(shortsword_idx)
    assert info["agent_attack_outcome"] == "Miss."
    assert env.agent.attacks_remaining == 0

def test_turn_does_not_end_with_attacks_remaining(scout_vs_wall_env):
    env = scout_vs_wall_env
    # Even with no speed and no bonus actions, a remaining attack slot keeps the turn alive.
    env.agent.used_action_this_turn = True
    env.agent.speed_remaining = 0
    env.agent.attacks_remaining = 1
    assert env._turn_economy_exhausted(env.agent) is False
    env.agent.attacks_remaining = 0
    assert env._turn_economy_exhausted(env.agent) is True

def test_long_range_disadvantage():
    tough_enemy = {"ac": 100, "max_hp": 50, "speed_total": 6, "attacks": [], "bonus_actions": []}
    env = DnDCombatEnv(map_width=40, map_height=40,
                       agent_stats=_scout_stats(), enemy_stats=tough_enemy,
                       render_mode=None)
    env.reset(seed=301)
    env.agent.position = [0, 0]
    env.enemy.position = [35, 0]  # beyond longbow normal range (30), within long (120)
    longbow_idx = find_action(env, "attack_longbow")

    obs, _, _, _, info = env.step(longbow_idx)
    assert "long range" in info.get("agent_attack_modifier", "").lower()
    assert info["agent_attack_outcome"] != "Out of range."

    # Beyond long range the attack is simply invalid
    env2 = DnDCombatEnv(map_width=100, map_height=100,
                        agent_stats=_scout_stats(), enemy_stats=tough_enemy,
                        render_mode=None)
    env2.reset(seed=302)
    env2.agent.position = [0, 0]
    env2.enemy.position = [99, 30]  # distance 129 > 120
    obs, _, _, _, info = env2.step(find_action(env2, "attack_longbow"))
    assert info["agent_attack_outcome"] == "Out of range."

def test_wolf_bite_knocks_prone():
    commoner_like = {"ac": 10, "max_hp": 10, "speed_total": 6,
                     "attacks": [], "bonus_actions": []}
    env = DnDCombatEnv(map_width=10, map_height=10,
                       agent_stats=_wolf_stats(), enemy_stats=commoner_like,
                       render_mode=None)
    env.reset(seed=303)
    env.agent.position = [0, 0]
    env.enemy.position = [0, 1]

    # Scripted dice: attack d20=20 (hit), damage=3, save d20=1 (1+0 < DC 11 -> fail)
    env.dice_roller = make_roller([20, 3, 1])
    obs, _, _, _, info = env.step(find_action(env, "attack_bite"))

    assert "Hit" in info["agent_attack_outcome"]
    assert env.enemy.has_condition("prone")
    assert "failed" in info["agent_rider_outcome"]
    assert obs["enemy_is_prone"][0] == 1.0

    # Same setup but the save succeeds: d20=11 (11+0 >= 11)
    env.reset(seed=304)
    env.agent.position = [0, 0]
    env.enemy.position = [0, 1]
    env.dice_roller = make_roller([20, 3, 11])
    obs, _, _, _, info = env.step(find_action(env, "attack_bite"))
    assert not env.enemy.has_condition("prone")
    assert "succeeded" in info["agent_rider_outcome"]

def test_prone_crawl_and_stand_up(dnd_env):
    dnd_env.reset(seed=305)
    agent = dnd_env.agent  # speed_total 6
    agent.position = [5, 5]
    dnd_env.enemy.position = [0, 0]  # far away, no OA interference

    move_e_idx = find_action(dnd_env, "move_E")
    stand_up_idx = find_action(dnd_env, "stand_up")

    # stand_up while not prone is an invalid action
    obs, reward, _, _, info = dnd_env.step(stand_up_idx)
    assert "not prone" in info["agent_action_details"]

    # Crawling while prone costs double
    agent.conditions.add("prone")
    speed_before = agent.speed_remaining
    obs, _, _, _, info = dnd_env.step(move_e_idx)
    assert agent.speed_remaining == speed_before - 2

    # Standing up costs half total speed and clears prone
    speed_before = agent.speed_remaining
    obs, _, _, _, info = dnd_env.step(stand_up_idx)
    assert not agent.has_condition("prone")
    assert agent.speed_remaining == speed_before - 3  # 6 // 2
    assert obs["agent_is_prone"][0] == 0.0

def test_observation_mirrors_conditions(dnd_env):
    dnd_env.reset(seed=306)
    dnd_env.agent.conditions.add("prone")
    dnd_env.enemy.is_dodging = True

    obs = dnd_env._get_obs()
    assert obs["agent_is_prone"][0] == 1.0
    assert obs["enemy_is_dodging"][0] == 1.0

    enemy_obs = dnd_env._get_enemy_observation()
    # From the enemy's perspective the roles flip
    assert enemy_obs["enemy_is_prone"][0] == 1.0
    assert enemy_obs["agent_is_dodging"][0] == 1.0
    assert set(enemy_obs.keys()) == set(obs.keys())

def test_hide_attack_reveals():
    # Goblin (Stealth +6) vs a watcher with passive Perception 10
    env = DnDCombatEnv(map_width=10, map_height=10,
                       agent_stats=_goblin_stats(), enemy_stats=dict(WATCHER),
                       render_mode=None)
    env.reset(seed=400)
    env.agent.position = [0, 0]
    env.enemy.position = [5, 5]

    # Stealth d20=5 -> 5+6=11 >= 10 -> hidden
    env.dice_roller = make_roller([5])
    obs, _, _, _, info = env.step(find_action(env, "hide"))
    assert env.agent.has_condition("hidden")
    assert obs["agent_is_hidden"][0] == 1.0
    assert "Hidden" in info["agent_hide_outcome"]

    # Hiding again while hidden is invalid (action already used anyway)
    obs, _, _, _, info = env.step(find_action(env, "hide"))
    assert "already used" in info["agent_action_details"] or "already hidden" in info["agent_action_details"]

    # Attacking from hiding: advantage on the roll, then revealed (hit or miss)
    env2 = DnDCombatEnv(map_width=10, map_height=10,
                        agent_stats=_goblin_stats(), enemy_stats=dict(WATCHER),
                        render_mode=None)
    env2.reset(seed=401)
    env2.agent.position = [0, 0]
    env2.enemy.position = [5, 5]
    env2.agent.conditions.add("hidden")
    # advantage roll (2d20kh1)=18 -> hit vs AC 10; damage=2
    env2.dice_roller = make_roller([18, 2])
    obs, _, _, _, info = env2.step(find_action(env2, "attack_shortbow"))
    assert "attacks from hiding, advantage" in info["agent_attack_modifier"]
    assert not env2.agent.has_condition("hidden")
    assert info["agent_hidden_revealed"] == "Revealed by attacking."

def test_hide_requires_distance_and_adjacency_reveals():
    env = DnDCombatEnv(map_width=10, map_height=10,
                       agent_stats=_goblin_stats(), enemy_stats=dict(WATCHER),
                       render_mode=None)
    env.reset(seed=402)
    env.agent.position = [0, 0]
    env.enemy.position = [0, 1]  # adjacent

    obs, _, _, _, info = env.step(find_action(env, "hide"))
    assert not env.agent.has_condition("hidden")
    assert "cannot hide while adjacent" in info["agent_action_details"]
    assert not env.agent.used_action_this_turn  # invalid attempt costs nothing

    # Hidden creature moving adjacent to its enemy is revealed
    env.reset(seed=403)
    env.agent.position = [0, 2]
    env.enemy.position = [0, 0]
    env.agent.conditions.add("hidden")
    move_n_idx = next(i for i, a in enumerate(env.action_map) if a["name"] == "move_N")
    obs, _, _, _, info = env.step(move_n_idx)  # to [0,1], adjacent to enemy
    assert not env.agent.has_condition("hidden")
    assert info["agent_hidden_revealed"] == "Revealed by adjacency."

def test_search_with_keen_senses():
    # Wolf seeker: Perception +3 and keen senses (advantage -> 2d20kh1)
    hider = {"ac": 10, "max_hp": 20, "speed_total": 6, "attacks": [], "bonus_actions": [],
             "skills": {"stealth": 6}}
    env = DnDCombatEnv(map_width=10, map_height=10,
                       agent_stats=_wolf_stats(), enemy_stats=hider,
                       render_mode=None)
    env.reset(seed=404)
    env.agent.position = [0, 0]
    env.enemy.position = [5, 5]
    env.enemy.conditions.add("hidden")

    # Perception (2d20kh1)=14 -> 14+3=17 vs DC 10+6=16 -> found
    env.dice_roller = make_roller([14])
    obs, _, _, _, info = env.step(find_action(env, "search"))
    assert not env.enemy.has_condition("hidden")
    assert "Found" in info["agent_search_outcome"]

    # Failed search leaves the target hidden
    env.reset(seed=405)
    env.agent.position = [0, 0]
    env.enemy.position = [5, 5]
    env.enemy.conditions.add("hidden")
    env.dice_roller = make_roller([10])  # 10+3=13 < 16
    obs, _, _, _, info = env.step(find_action(env, "search"))
    assert env.enemy.has_condition("hidden")
    assert "failed" in info["agent_search_outcome"].lower()

    # Searching with nothing hidden is an invalid action
    env.reset(seed=406)
    env.enemy.conditions.discard("hidden")
    obs, _, _, _, info = env.step(find_action(env, "search"))
    assert "nothing is hidden" in info["agent_action_details"]
    assert not env.agent.used_action_this_turn

def test_nimble_escape_bonus_actions():
    env = DnDCombatEnv(map_width=10, map_height=10,
                       agent_stats=_goblin_stats(), enemy_stats=dict(WATCHER),
                       render_mode=None)
    names = [a["name"] for a in env.action_map]
    assert "bonus_disengage" in names and "bonus_hide" in names

    # Creatures without the trait don't get the actions
    env_plain = DnDCombatEnv(map_width=10, map_height=10,
                             agent_stats=dict(WATCHER), enemy_stats=dict(WATCHER),
                             render_mode=None)
    plain_names = [a["name"] for a in env_plain.action_map]
    assert "bonus_disengage" not in plain_names and "bonus_hide" not in plain_names

    env.reset(seed=407)
    env.agent.position = [0, 0]
    env.enemy.position = [0, 1]

    # Bonus-action Disengage costs the bonus action, not the Action
    obs, _, _, _, info = env.step(find_action(env, "bonus_disengage"))
    assert env.agent.is_disengaging
    assert env.agent.used_bonus_action_this_turn
    assert not env.agent.used_action_this_turn

    # A second bonus action the same turn is refused
    obs, _, _, _, info = env.step(find_action(env, "bonus_hide"))
    assert "already used" in info["agent_action_details"]

    # bonus_hide while adjacent is invalid (fresh turn)
    env.reset(seed=420)
    env.agent.position = [0, 0]
    env.enemy.position = [0, 1]
    obs, _, _, _, info = env.step(find_action(env, "bonus_hide"))
    assert "cannot hide while adjacent" in info["agent_action_details"]
    assert not env.agent.used_bonus_action_this_turn  # invalid attempt costs nothing

    # bonus_hide at distance works and leaves the Action free
    env.reset(seed=408)
    env.agent.position = [0, 0]
    env.enemy.position = [5, 5]
    env.dice_roller = make_roller([5])  # 5+6=11 >= 10 -> hidden
    obs, _, _, _, info = env.step(find_action(env, "bonus_hide"))
    assert env.agent.has_condition("hidden")
    assert env.agent.used_bonus_action_this_turn
    assert not env.agent.used_action_this_turn

def test_shove_prone_and_push():
    # Guard (STR 13 -> +1) vs watcher (best of STR/DEX -> +0)
    env = DnDCombatEnv(map_width=10, map_height=10,
                       agent_stats=_guard_stats(), enemy_stats=dict(WATCHER),
                       render_mode=None)
    env.reset(seed=409)
    env.agent.position = [0, 0]
    env.enemy.position = [0, 1]

    # Contest: initiator d20=15 (+1)=16 vs defender d20=5 (+0)=5 -> win
    env.dice_roller = make_roller([15, 5])
    obs, _, _, _, info = env.step(find_action(env, "shove_prone"))
    assert env.enemy.has_condition("prone")
    assert env.agent.used_action_this_turn and env.agent.attacks_remaining == 0
    assert "prone" in info["agent_contest_outcome"]

    # Push: enemy shoved one cell directly away
    env.reset(seed=410)
    env.agent.position = [0, 0]
    env.enemy.position = [0, 1]
    env.dice_roller = make_roller([15, 5])
    obs, _, _, _, info = env.step(find_action(env, "shove_push"))
    assert env.enemy.position == [0, 2]

    # Push against the map edge: contest won but nowhere to go
    env.reset(seed=411)
    env.agent.position = [0, 8]
    env.enemy.position = [0, 9]
    env.dice_roller = make_roller([15, 5])
    obs, _, _, _, info = env.step(find_action(env, "shove_push"))
    assert env.enemy.position == [0, 9]
    assert "nowhere to go" in info["agent_contest_outcome"]

    # Losing the contest changes nothing
    env.reset(seed=412)
    env.agent.position = [0, 0]
    env.enemy.position = [0, 1]
    env.dice_roller = make_roller([5, 15])
    obs, _, _, _, info = env.step(find_action(env, "shove_prone"))
    assert not env.enemy.has_condition("prone")
    assert "failed" in info["agent_contest_outcome"].lower()

    # Shove out of reach is invalid and costs nothing
    env.reset(seed=413)
    env.agent.position = [0, 0]
    env.enemy.position = [5, 5]
    obs, _, _, _, info = env.step(find_action(env, "shove_prone"))
    assert "out of reach" in info["agent_action_details"]
    assert not env.agent.used_action_this_turn

def test_grapple_escape_and_auto_release():
    env = DnDCombatEnv(map_width=10, map_height=10,
                       agent_stats=_guard_stats(), enemy_stats=dict(WATCHER),
                       render_mode=None)
    env.reset(seed=414)
    env.agent.position = [0, 0]
    env.enemy.position = [0, 1]

    # Grapple: initiator 15+1 vs defender 5+0 -> enemy grappled
    env.dice_roller = make_roller([15, 5])
    obs, _, _, _, info = env.step(find_action(env, "grapple"))
    assert env.enemy.has_condition("grappled")
    assert obs["enemy_is_grappled"][0] == 1.0

    # A grappled creature cannot move
    move_info = {}
    _, _, moved = env._execute_micro_action(
        env.enemy, env.agent, {"type": "move", "delta": (0, 1), "name": "move_S"}, move_info, 'enemy')
    assert moved is False
    assert move_info["enemy_move_outcome"] == "Cannot move while grappled."

    # Escaping: escaper best-of(STR/DEX) +0 vs grappler Athletics +1; 18 vs 2 -> free
    env.dice_roller = make_roller([18, 2])
    escape_info = {}
    _, _, escaped = env._execute_micro_action(
        env.enemy, env.agent, {"type": "escape_grapple", "name": "escape_grapple"}, escape_info, 'enemy')
    assert escaped is True
    assert not env.enemy.has_condition("grappled")
    assert "Escaped" in escape_info["enemy_contest_outcome"]

    # Auto-release: grappler walking out of reach frees the target
    env.reset(seed=415)
    env.agent.position = [0, 1]
    env.enemy.position = [0, 2]
    env.enemy.conditions.add("grappled")
    move_n_idx = next(i for i, a in enumerate(env.action_map) if a["name"] == "move_N")
    obs, _, _, _, info = env.step(move_n_idx)  # agent to [0,0], distance now 2
    assert not env.enemy.has_condition("grappled")
    assert "out of reach" in info["enemy_grapple_released"]

def test_rule_based_enemy_searches_for_hidden_agent():
    env = DnDCombatEnv(map_width=10, map_height=10,
                       agent_stats=_goblin_stats(), enemy_stats=_wolf_stats(),
                       render_mode=None)
    env.reset(seed=416)
    env.agent.position = [0, 0]
    env.enemy.position = [9, 9]
    env.agent.conditions.add("hidden")

    end_idx = next(i for i, a in enumerate(env.action_map) if a["type"] == "end_turn")
    obs, _, _, _, info = env.step(end_idx)
    # Wolf spent its action searching (keen senses give it advantage)
    assert "enemy_search_outcome" in info or not env.agent.has_condition("hidden")
