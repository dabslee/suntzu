import pytest
import numpy as np
import os # For skipping human render test in headless CI
from gymnasium import spaces 
from dnd_gym_env import Creature, DnDCombatEnv 
from dice import roll 
from bestiary.commoner import get_commoner_stats # For render tests
from bestiary.wolf import get_wolf_stats     # For render tests

# --- Test Creature Class ---

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


# --- Test DnDCombatEnv Class ---

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

def test_env_initialization(dnd_env, agent_stats_env, enemy_stats_env):
    assert dnd_env.agent is not None
    assert dnd_env.enemy is not None
    assert dnd_env.agent.name == "agent"
    assert dnd_env.enemy.name == "enemy"
    assert dnd_env.map_width == 10 and dnd_env.map_height == 10

    # Action space: 4 moves + num_agent_attacks + num_agent_bonus_actions + 1 pass
    expected_num_actions = 4 + len(agent_stats_env["attacks"]) + len(agent_stats_env["bonus_actions"]) + 1
    assert isinstance(dnd_env.action_space, spaces.Discrete)
    assert dnd_env.action_space.n == expected_num_actions

    # Observation space
    assert isinstance(dnd_env.observation_space, spaces.Dict)
    obs_keys = ["agent_hp_norm", "enemy_hp_norm", "agent_pos", "enemy_pos", "distance_to_enemy", "agent_speed_remaining_norm"]
    for key in obs_keys:
        assert key in dnd_env.observation_space.spaces
    assert dnd_env.observation_space["agent_hp_norm"].shape == (1,)
    assert dnd_env.observation_space["agent_pos"].shape == (2,)


def test_reset_env(dnd_env):
    obs, info = dnd_env.reset(seed=42)
    
    assert dnd_env.agent.current_hp == dnd_env.agent.max_hp
    assert dnd_env.enemy.current_hp == dnd_env.enemy.max_hp
    assert dnd_env.agent.is_alive
    assert dnd_env.enemy.is_alive
    
    assert 0 <= dnd_env.agent.position[0] < dnd_env.map_width
    assert 0 <= dnd_env.agent.position[1] < dnd_env.map_height
    assert 0 <= dnd_env.enemy.position[0] < dnd_env.map_width
    assert 0 <= dnd_env.enemy.position[1] < dnd_env.map_height
    assert dnd_env.agent.position != dnd_env.enemy.position

    assert obs["agent_hp_norm"][0] == 1.0
    assert obs["enemy_hp_norm"][0] == 1.0
    assert np.array_equal(obs["agent_pos"], dnd_env.agent.position)


def test_step_agent_attack_enemy_ko(dnd_env, agent_stats_env):
    # Setup: Place agent next to enemy, make enemy very weak
    dnd_env.reset(seed=1) # Use a seed for predictable placement if needed
    dnd_env.agent.position = [0,0]
    dnd_env.enemy.position = [0,1] # Adjacent
    dnd_env.enemy.current_hp = 1 # Enemy has 1 HP
    dnd_env.enemy.max_hp = 1 
    dnd_env.agent.attacks[0]['damage_dice'] = "1d4+50" # Guaranteed KO if hit

    attack_action_idx = -1
    for i, action_def in enumerate(dnd_env.action_map):
        if action_def["type"] == "attack":
            attack_action_idx = i
            break
    assert attack_action_idx != -1

    # Ensure agent hits (AC of enemy is 13, agent to_hit is +5, needs ~8+ on d20)
    # For deterministic test, we might need to mock dice roll or set AC extremely low
    # For now, rely on high damage dice to KO if a hit occurs.
    # If this test is flaky, dice mocking for the attack roll is the next step.
    
    # Calculate expected damage: enemy has 1 HP, so damage dealt is 1.
    # If make_attack was mocked to always hit and deal exact damage, this would be more precise.
    # For now, we assume the +50 damage ensures a KO. The actual damage_dealt for reward will be 1.
    # The damage dealt component of reward is based on actual HP lost.
    damage_to_ko_enemy = dnd_env.enemy.current_hp 
    
    obs, reward, terminated, truncated, info = dnd_env.step(attack_action_idx)
    
    if dnd_env.enemy.is_alive:
        # This might happen if the attack roll (1d20 + to_hit) misses.
        # For this test to be robust, we need a guaranteed hit or mock dice.
        # Given the high damage_dice, a hit is almost certain to KO.
        pytest.skip("Agent attack missed in KO test, cannot verify precise reward. Consider dice mocking for robustness.")

    assert terminated is True, "Episode should be terminated after enemy KO"
    assert dnd_env.enemy.is_alive is False, "Enemy should be KO'd"
    
    # Expected reward: -0.1 (step) + damage_to_ko_enemy (actual_hp_lost) + 100 (win)
    # Survival reward is not added here because the agent's action terminated the episode.
    expected_reward = -0.1 + damage_to_ko_enemy + 100
    assert reward == pytest.approx(expected_reward), f"Reward for KOing enemy is incorrect. Got {reward}, expected {expected_reward}"
    assert obs["enemy_hp_norm"][0] == 0.0


def test_step_enemy_attack_agent_ko(dnd_env, enemy_stats_env): # agent_stats_env is implicitly used by dnd_env fixture
    dnd_env.reset(seed=2)
    dnd_env.agent.position = [0,0]
    dnd_env.enemy.position = [0,1] # Adjacent
    
    # Agent has 1 HP, so damage taken to KO is 1.
    dnd_env.agent.current_hp = 1 
    dnd_env.agent.max_hp = 1 # Setting max_hp to 1 for consistency in this test
    
    # Ensure enemy attack is lethal enough if it hits.
    # The default scimitar "1d6+2" is likely to be >1.
    # No need to change enemy_stats_env['attacks'][0]['damage_dice'] if it's already strong enough.
    # damage_taken_to_ko_agent = dnd_env.agent.current_hp # which is 1

    # Agent passes turn, allowing enemy to act
    pass_action_idx = -1
    for i, action_def in enumerate(dnd_env.action_map):
        if action_def["type"] == "pass":
            pass_action_idx = i
            break
    assert pass_action_idx != -1
    
    obs, reward, terminated, truncated, info = dnd_env.step(pass_action_idx)

    # If agent is still alive, it means the enemy's attack missed.
    if dnd_env.agent.is_alive:
        pytest.skip("Enemy attack missed agent in KO test, cannot verify precise reward. Consider dice mocking for robustness.")

    assert terminated is True, "Episode should be terminated after agent KO"
    assert dnd_env.agent.is_alive is False, "Agent should be KO'd"
    
    # Expected reward components:
    # -0.1 (step penalty for agent's pass action)
    # +0.5 (agent survival after its pass action)
    # -1 (damage taken by agent, since agent_hp was 1)
    # -100 (loss penalty)
    # Total = -0.1 + 0.5 - 1 - 100 = -100.6
    # The damage_taken_by_agent is implicitly 1 because current_hp was 1.
    damage_taken_to_ko_agent = 1 # Based on setup current_hp=1
    expected_reward = -0.1 + 0.5 - damage_taken_to_ko_agent - 100
    
    assert reward == pytest.approx(expected_reward), f"Reward for agent KO is incorrect. Got {reward}, expected {expected_reward}"
    assert obs["agent_hp_norm"][0] == 0.0


def test_step_movement(dnd_env):
    obs_initial, _ = dnd_env.reset(seed=3)
    initial_pos = list(obs_initial["agent_pos"]) # Ensure it's a list for comparison

    # Find a valid move action (e.g., move East)
    move_action_idx = -1
    expected_delta = None
    for i, action_def in enumerate(dnd_env.action_map):
        if action_def["type"] == "move" and action_def["name"] == "move_E": # East
            # Check if move is possible from initial_pos
            if initial_pos[0] < dnd_env.map_width - 1:
                move_action_idx = i
                expected_delta = action_def["delta"]
                break
    
    if move_action_idx == -1: # If agent is at the right edge, try another direction
        for i, action_def in enumerate(dnd_env.action_map):
            if action_def["type"] == "move" and action_def["name"] == "move_N": # North
                 if initial_pos[1] > 0: # Check if move North is possible
                    move_action_idx = i
                    expected_delta = action_def["delta"]
                    break
    
    assert move_action_idx != -1, "Could not find a valid move action for test_step_movement"
    assert expected_delta is not None

    obs, reward, terminated, truncated, info = dnd_env.step(move_action_idx)
    
    expected_new_pos = [initial_pos[0] + expected_delta[0], initial_pos[1] + expected_delta[1]]
    
    assert np.array_equal(obs["agent_pos"], expected_new_pos)
    # Expected reward: -0.1 (step penalty) + 0.5 (survival reward) = 0.4
    assert reward == pytest.approx(0.4), "Reward for simple move action is incorrect"
    assert terminated is False

def test_agent_move_into_occupied_cell(dnd_env):
    dnd_env.reset(seed=6)
    # Place agent and enemy adjacent
    dnd_env.agent.position = [0, 0]
    dnd_env.enemy.position = [0, 1] # Enemy to the South
    initial_agent_speed = dnd_env.agent.speed_total
    dnd_env.agent.speed_remaining = initial_agent_speed

    # Find action for agent to move South (into enemy)
    move_south_action_idx = -1
    for i, action_def in enumerate(dnd_env.action_map):
        if action_def["type"] == "move" and action_def["name"] == "move_S":
            move_south_action_idx = i
            break
    assert move_south_action_idx != -1

    obs, reward, terminated, truncated, info = dnd_env.step(move_south_action_idx)

    assert np.array_equal(obs["agent_pos"], [0, 0]), "Agent should not have moved"
    # Agent attempted to move 1 cell, so 1 speed point should be consumed.
    assert dnd_env.agent.speed_remaining == initial_agent_speed - 1, "Agent speed not deducted correctly for failed move"
    assert terminated is False

def test_enemy_move_into_occupied_cell(dnd_env, enemy_stats_env): # Add enemy_stats_env fixture
    dnd_env.reset(seed=7)
    
    # Modify enemy to have no attacks for this test, to force movement attempt
    original_enemy_attacks = dnd_env.enemy.attacks
    dnd_env.enemy.attacks = []

    # Place agent and enemy adjacent
    dnd_env.agent.position = [0, 1] 
    dnd_env.enemy.position = [0, 0] # Enemy North of Agent, wants to move to [0,1] (agent's spot)
    initial_enemy_speed = dnd_env.enemy.speed_total
    dnd_env.enemy.speed_remaining = initial_enemy_speed # Ensure enemy has speed


    # Agent passes, so enemy gets to move
    pass_action_idx = -1
    for i, action_def in enumerate(dnd_env.action_map):
        if action_def["type"] == "pass":
            pass_action_idx = i
            break
    assert pass_action_idx != -1
    
    obs, reward, terminated, truncated, info = dnd_env.step(pass_action_idx)

    # Enemy AI: if adjacent and no attacks, it should attempt to move.
    # If it had attacks, it would attack. Since no attacks, it should try to move.
    # The default AI moves if not adjacent OR if it cannot attack.
    # Current AI: if dist_to_agent == 1: if self.enemy.attacks: ATTACK else: (no explicit move)
    # This means if adjacent and no attacks, it does nothing.
    # We need to ensure the "else (not adjacent)" part of AI is triggered, OR that the "adjacent" part tries to move if no attacks.

    # Let's adjust AI slightly for this test or setup:
    # For this test, we need the enemy to try to move into agent's cell.
    # The AI logic is: if adjacent, attack. Else, move.
    # If attacks are empty, it won't attack. It will then fall through and NOT move if adjacent.
    # This test is still problematic for the current AI.

    # Re-evaluate: The code for enemy movement (dx, dy towards agent) has paths for collision.
    # if [enemy_target_x_try1, enemy_target_y_try1] == self.agent.position:
    #    self.enemy.speed_remaining -= enemy_move_cost
    # This code is in the `else` (not adjacent) block of the enemy AI.
    # So, for this test to be effective, the enemy MUST NOT be adjacent.

    # New setup: Enemy at [0,0], Agent at [0,2]. Enemy wants to move to [0,1]. This is not occupied.
    # This test should be: enemy wants to move to [0,1], but agent is ALREADY at [0,1]. Enemy starts at [0,0].
    # This makes them adjacent. The AI will attack if it can.
    # If enemy has no attacks, it does nothing if adjacent.

    # The simplest way to test the "enemy move blocked by agent" code path
    # is to ensure the enemy is NOT adjacent, but its desired single step (e.g. dx part)
    # would land on the agent.
    
    dnd_env.agent.position = [1,0] # Agent at [1,0]
    dnd_env.enemy.position = [0,0] # Enemy at [0,0], wants to move to dx=1 -> [1,0]

    # Now dist is 1. Enemy AI with no attacks will do nothing.
    # We need to modify the AI or the test.
    # Let's assume for this test, enemy AI always tries to move if it cannot attack.
    # This requires a small temporary modification to the environment's AI logic,
    # or more simply, accept that this specific code path in enemy AI (move into agent spot)
    # is hard to test without AI modification if it prioritizes attacks/does nothing else.

    # For now, let's assume the test setup was trying to verify the collision code itself,
    # not the complex interaction with AI decision-making if AI avoids the situation.
    # The previous run failed because speed was not deducted. This means the enemy did not attempt the move.
    # This was because it was adjacent and attacked (or would have if it had attacks).
    # If we remove its attacks, and it's adjacent, current AI does nothing.

    # The fix for the failed test is that the enemy AI logic needs to be changed
    # OR the test needs to be more creative.
    # The existing code for speed deduction on collision IS present in the dnd_gym_env.
    # The failure is that this code path wasn't hit by the AI.

    # Let's restore original attacks for now and acknowledge this test needs refinement
    # if the goal is to test the enemy AI's specific collision path.
    # The assertion failure (5 == 4) means speed was NOT deducted.
    # This is correct if the enemy AI decided to attack or do nothing instead of moving.
    
    # If enemy attacked, its speed is reset at start of its turn, then unchanged by attack.
    # If enemy had no attacks and was adjacent, it did nothing, speed unchanged.
    # The test is flawed because it assumes the enemy *will* attempt to move into the agent's square.

    # To make the test pass with current AI, we'd assert speed is UNCHANGED if adjacent.
    # But that's not testing "move into occupied cell".
    # Let's force non-adjacency where the target *would* be the agent cell.
    dnd_env.enemy.attacks = [] # No attacks
    dnd_env.agent.position = [1, 0] # Agent
    dnd_env.enemy.position = [-1, 0] # Enemy, wants to move to [0,0], then to [1,0] (agent)
                                    # This is too complex for current 1-step AI.

    # Simplest: Enemy at [0,0], Agent at [1,0]. Enemy has NO attacks.
    # Current AI: dist=1. Has attacks? No. Does nothing. Speed not deducted. Test fails.
    # This test was previously skipped due to AI complexity.
    # New approach: Enemy has no attacks, is NOT adjacent, but its path is to the agent's square.
    dnd_env.enemy.attacks = [] # Ensure enemy must move if it wants to act

    # Enemy at [0,0], Agent at [0,1]. Enemy wants to move South to [0,1].
    # This makes them non-adjacent for the AI's "attack first" logic if dist=1.
    # Enemy AI: if dist != 1, it will try to move.
    # Let's set them up so dist > 1, but one step of enemy move targets agent's square.
    # No, this is not right. The collision logic in enemy AI is for when its target dx/dy lands on agent.
    # The AI calculates dx, dy to agent. Then calculates target_x, target_y for its step.
    # This target_x, target_y is then checked against agent.position.

    dnd_env.agent.position = [0, 1]  # Agent
    dnd_env.enemy.position = [0, 0]  # Enemy one step North of Agent
    # Enemy AI will determine dx=0, dy=1. Target will be [0,1] (Agent's spot).
    # Since enemy has no attacks, its "if self.enemy.attacks:" check (if dist==1) fails.
    # The current AI when dist==1 and no attacks will do nothing.
    
    # To test the collision:
    # 1. Enemy must not be adjacent (so it enters move logic)
    # 2. Its calculated 1-step move (e.g. dx_component, then dy_component) must land on agent.
    
    dnd_env.agent.position = [1, 2]    # Agent
    dnd_env.enemy.position = [0, 2]    # Enemy West of Agent, dist = 1
                                       # AI: dx=1, dy=0. Target [1,2]
    # This is still adjacent. The AI will not move if it has no attacks and is adjacent.

    # Let's make enemy non-adjacent, but its first move component (e.g. x-axis) lands on agent
    dnd_env.agent.position = [1,0]     # Agent
    dnd_env.enemy.position = [0,0]     # Enemy. Wants to move to [1,0] (dx=1).
    initial_enemy_pos = list(dnd_env.enemy.position)
    initial_enemy_speed = dnd_env.enemy.speed_total
    dnd_env.enemy.speed_remaining = initial_enemy_speed
    
    # This setup is dist=1. If enemy has no attacks, it does nothing.
    # If enemy HAS attacks, it attacks.
    # The collision code for enemy move is in the `else` (not adjacent) part of AI.

    # New strategy: Enemy is not adjacent, but its path (x-component) is blocked by agent.
    dnd_env.agent.position = [1,0]  # Agent
    dnd_env.enemy.position = [0,0]  # Enemy at [0,0]
    dnd_env.enemy.attacks = []      # Enemy has no attacks, must move.
    
    # Manually set enemy further away so its move logic gets triggered
    dnd_env.enemy.position = [-1, 0] # Enemy is at [-1,0], agent at [1,0]. Dist = 2.
                                     # Enemy wants dx=1 (to [0,0]), then dx=1 again (to [1,0])
    initial_enemy_pos_far = list(dnd_env.enemy.position)
    initial_enemy_speed = dnd_env.enemy.speed_total # Should be 5 from fixture
    dnd_env.enemy.speed_remaining = initial_enemy_speed

    # Agent passes
    obs, reward, terminated, truncated, info = dnd_env.step(pass_action_idx)
    
    # Enemy's first move: dx=1, dy=0. New pos should be [0,0]. Speed should be initial_enemy_speed - 1
    assert np.array_equal(dnd_env.enemy.position, [0,0]), "Enemy should have moved one step closer"
    assert dnd_env.enemy.speed_remaining == initial_enemy_speed - 1 

    # Now enemy is at [0,0], agent at [1,0]. Enemy speed is initial_enemy_speed - 1.
    # Agent passes again
    obs, reward, terminated, truncated, info = dnd_env.step(pass_action_idx)

    # Enemy's second move: dx=1, dy=0. Target is [1,0] (agent's position).
    # This move should be blocked by the agent. Enemy position remains [0,0]. 
    # Speed was reset to initial_enemy_speed at the start of this (enemy's second) turn, then 1 is deducted for collision.
    assert np.array_equal(dnd_env.enemy.position, [0,0]), "Enemy should not have moved (blocked by agent)"
    print(f"DEBUG_TEST: Enemy speed before final assertion: {dnd_env.enemy.speed_remaining}, expected: {initial_enemy_speed - 1}")
    assert dnd_env.enemy.speed_remaining == initial_enemy_speed - 1, "Enemy speed not deducted correctly for collision in the second turn"
    assert terminated is False
    
    dnd_env.enemy.attacks = original_enemy_attacks # Restore attacks


def test_agent_move_into_wall_consumes_speed(dnd_env):
    dnd_env.reset(seed=8)
    # Place agent at an edge, e.g., North edge
    dnd_env.agent.position = [0, 0]
    initial_agent_speed = dnd_env.agent.speed_total
    dnd_env.agent.speed_remaining = initial_agent_speed

    # Find action for agent to move North (into wall)
    move_north_action_idx = -1
    for i, action_def in enumerate(dnd_env.action_map):
        if action_def["type"] == "move" and action_def["name"] == "move_N":
            move_north_action_idx = i
            break
    assert move_north_action_idx != -1

    obs, reward, terminated, truncated, info = dnd_env.step(move_north_action_idx)

    assert np.array_equal(obs["agent_pos"], [0, 0]), "Agent should not have moved"
    assert dnd_env.agent.speed_remaining == initial_agent_speed - 1, "Agent speed not deducted for bumping wall"
    assert terminated is False


def test_observation_space_content(dnd_env):
    obs, _ = dnd_env.reset(seed=4)
    
    obs_space = dnd_env.observation_space
    assert isinstance(obs, dict)
    
    for key, space in obs_space.spaces.items():
        assert key in obs, f"Key {key} missing in observation"
        assert isinstance(obs[key], np.ndarray), f"Observation for {key} is not a numpy array"
        assert obs[key].shape == space.shape, f"Shape mismatch for {key}"
        assert obs[key].dtype == space.dtype, f"Dtype mismatch for {key}"
        
        # Check values within bounds
        if isinstance(space, spaces.Box):
            assert np.all(obs[key] >= space.low), f"Value for {key} below lower bound"
            assert np.all(obs[key] <= space.high), f"Value for {key} above upper bound"

    # Test the specific agent_speed_remaining_norm clamping (addressed in dnd_gym_env.py)
    dnd_env.agent.speed_remaining = dnd_env.agent.speed_total + 5 # Force speed remaining > total
    clamped_obs = dnd_env._get_obs()
    assert clamped_obs["agent_speed_remaining_norm"][0] <= 1.0, \
        "agent_speed_remaining_norm not clamped at 1.0"
    assert clamped_obs["agent_speed_remaining_norm"][0] >= 0.0


def test_action_decoding_and_mapping(dnd_env):
    num_actions = dnd_env.action_space.n
    action_map = dnd_env.action_map # Access through property

    assert len(action_map) == num_actions

    for i in range(num_actions):
        decoded_action = dnd_env._decode_action(i) # Test the internal helper
        assert decoded_action == action_map[i]
        assert "type" in decoded_action
        assert "name" in decoded_action

        if decoded_action["type"] == "move":
            assert "delta" in decoded_action
        elif decoded_action["type"] == "attack":
            assert "index" in decoded_action
        elif decoded_action["type"] == "bonus_action":
            assert "index" in decoded_action # or "name" is enough
        elif decoded_action["type"] == "pass":
            pass # No other keys needed
        else:
            pytest.fail(f"Unknown decoded action type: {decoded_action['type']}")


def test_rendering_ansi(dnd_env): # dnd_env fixture has render_mode=None by default
    dnd_env.render_mode = 'ansi' # Explicitly set to ansi for this test
    dnd_env.reset(seed=5)
    try:
        # The current render prints to stdout and returns the string.
        # We can capture stdout or just check if it runs and returns string.
        render_output = dnd_env.render()
        assert isinstance(render_output, str)
        assert "Agent HP" in render_output
        assert "Enemy HP" in render_output
        # Check for agent 'A' and enemy 'E' (if alive)
        if dnd_env.agent.is_alive: assert 'A' in render_output
        if dnd_env.enemy.is_alive: assert 'E' in render_output

    except Exception as e:
        pytest.fail(f"env.render() raised an exception: {e}")


def test_episode_truncation(dnd_env): # Uses the dnd_env fixture
    env = dnd_env 
    obs, info = env.reset(seed=12) # Use a seed for reproducibility
    assert env.current_episode_steps == 0, "current_episode_steps should be 0 after reset"

    # Determine a pass action.
    pass_action_idx = -1
    for i, action_def in enumerate(env.action_map):
        if action_def.get("name") == "pass_turn": # Assuming name is "pass_turn"
            pass_action_idx = i
            break
    if pass_action_idx == -1:
        pytest.fail("Could not determine 'pass_turn' action from env.action_map.")

    terminated = False
    truncated = False # Variable to store truncated from last step
    
    # Loop up to one step before max_episode_steps
    # env.max_episode_steps should be 100 as per DnDCombatEnv.__init__
    for step_num_one_based in range(1, env.max_episode_steps): 
        assert env.current_episode_steps == step_num_one_based - 1, \
            f"Step count mismatch before step {step_num_one_based}. Expected {step_num_one_based - 1}, got {env.current_episode_steps}"
        
        # Store the flags from the step call
        obs, reward, terminated_step, truncated_step, info = env.step(pass_action_idx)
        terminated = terminated_step # Update loop control based on this step
        truncated = truncated_step   # Update loop control based on this step
        
        assert env.current_episode_steps == step_num_one_based, \
            f"Step count mismatch after step {step_num_one_based}. Expected {step_num_one_based}, got {env.current_episode_steps}"
        assert not terminated, \
            f"Episode should not terminate at step {step_num_one_based} ({env.current_episode_steps}/{env.max_episode_steps}) with only pass actions"
        assert not truncated, \
            f"Episode should not truncate at step {step_num_one_based} ({env.current_episode_steps}/{env.max_episode_steps}) with only pass actions"

    # On the max_episode_steps-th step (this is the step that should cause truncation)
    assert env.current_episode_steps == env.max_episode_steps - 1, \
        f"Should be at step {env.max_episode_steps -1} before the truncating step. Got {env.current_episode_steps}"
    
    obs, reward, terminated_step, truncated_step, info = env.step(pass_action_idx)
    terminated = terminated_step
    truncated = truncated_step
    
    assert env.current_episode_steps == env.max_episode_steps, \
        f"Step count should be {env.max_episode_steps} at truncation. Got {env.current_episode_steps}"
    assert not terminated, "Episode should not be 'terminated' on max_steps if only pass actions lead to truncation"
    assert truncated is True, "Episode should be 'truncated' on max_steps"
    
    # Check if current_episode_steps resets after truncation
    obs, info = env.reset(seed=13) # Use a different seed for reset
    assert env.current_episode_steps == 0, "current_episode_steps should be 0 after reset following a truncated episode"
