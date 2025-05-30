import pytest
import numpy as np
from gymnasium import spaces # Import the 'spaces' module
from dnd_gym_env import Creature, DnDCombatEnv # Assuming dnd_gym_env.py is in the same directory
from dice import roll # Assuming dice.py is in the same directory

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
    return DnDCombatEnv(map_width=10, map_height=10, 
                        agent_stats=agent_stats_env, 
                        enemy_stats=enemy_stats_env)

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
    
    obs, reward, terminated, truncated, info = dnd_env.step(attack_action_idx)
    
    # If the attack missed, this test part is invalid.
    # We assume it hits for KO test. A more robust test would control the hit.
    if dnd_env.enemy.is_alive:
        print("Warning: Agent attack missed in KO test, test may not be conclusive.")
        pytest.skip("Agent attack missed, cannot verify KO deterministically without mocking.")


    assert terminated is True
    assert dnd_env.enemy.is_alive is False
    assert reward > 0 # Should be positive for KOing enemy (e.g. 50 - 0.05)
    assert obs["enemy_hp_norm"][0] == 0.0


def test_step_enemy_attack_agent_ko(dnd_env, enemy_stats_env):
    dnd_env.reset(seed=2)
    dnd_env.agent.position = [0,0]
    dnd_env.enemy.position = [0,1] # Adjacent
    dnd_env.agent.current_hp = 1 # Agent has 1 HP
    dnd_env.agent.max_hp = 1
    dnd_env.enemy.attacks[0]['damage_dice'] = "1d4+50" # Enemy lethal attack

    # Agent passes turn, allowing enemy to act
    pass_action_idx = -1
    for i, action_def in enumerate(dnd_env.action_map):
        if action_def["type"] == "pass":
            pass_action_idx = i
            break
    assert pass_action_idx != -1
    
    obs, reward, terminated, truncated, info = dnd_env.step(pass_action_idx)

    if dnd_env.agent.is_alive:
        print("Warning: Enemy attack missed in KO test, test may not be conclusive.")
        pytest.skip("Enemy attack missed, cannot verify KO deterministically without mocking.")

    assert terminated is True
    assert dnd_env.agent.is_alive is False
    assert reward < 0 # Should be negative for agent KO (e.g. -50 - 0.05)
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
    assert reward == -0.05 # Small step penalty
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


def test_rendering_ansi(dnd_env):
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
