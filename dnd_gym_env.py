from typing import List, Dict, Callable, Tuple, Optional, Any, Union
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os 

# Pygame import (optional)
try:
    import pygame
except ImportError:
    pygame = None # Flag to indicate Pygame is not available

# SB3 and FlattenObservation imports (optional)
try:
    from stable_baselines3 import DQN
    from gymnasium.wrappers import FlattenObservation
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    # Define dummy classes if SB3 not available for type hints and class structure
    class DQN: pass 
    class FlattenObservation: pass
    print("Warning: stable-baselines3 or gymnasium.wrappers not found. DRL agent functionality will be limited.")


# Assuming dice.py is in the same directory or PYTHONPATH
try:
    from dice import roll
except ImportError:
    # Fallback for environments where dice.py might not be directly available
    # This allows the file to be parsed, but it will fail at runtime if roll is used.
    print("Warning: dice.py not found. DnDCombatEnv will not work without it.")
    def roll(dice_string: str) -> int:
        raise NotImplementedError("Dice roller not available.")

class Creature:
    """
    Represents a creature in a D&D-like environment.
    """
    def __init__(self, 
                 name: str, 
                 ac: int, 
                 max_hp: int, 
                 speed_total: int, 
                 attacks: List[Dict], 
                 bonus_actions: List[str], 
                 initial_position: List[int]):
        self.name: str = name
        self.ac: int = ac
        self.max_hp: int = max_hp
        self.current_hp: int = max_hp
        self.speed_total: int = speed_total # Speed in grid cells
        self.speed_remaining: int = speed_total
        
        for attack in attacks:
            if not all(k in attack for k in ["name", "to_hit", "damage_dice", "num_attacks"]):
                raise ValueError("Each attack dictionary must contain 'name', 'to_hit', 'damage_dice', and 'num_attacks'.")
        self.attacks: List[Dict] = attacks
        
        self.bonus_actions: List[str] = bonus_actions
        
        if len(initial_position) != 2:
            raise ValueError("initial_position must be a list of two integers [x, y].")
        self.position: List[int] = list(initial_position)
        
        self.is_alive: bool = self.current_hp > 0

        # Combat state attributes
        self.used_action_this_turn: bool = False
        self.used_bonus_action_this_turn: bool = False
        self.used_reaction_this_round: bool = False # Note: Reactions are typically once per ROUND, not turn.
        self.is_dodging: bool = False
        self.is_disengaging: bool = False
        self.took_dash_action_this_turn: bool = False


    def take_damage(self, amount: int) -> None:
        if amount < 0: amount = 0
        self.current_hp -= amount
        if self.current_hp < 0: self.current_hp = 0
        if self.current_hp <= 0: self.is_alive = False

    def make_attack(self, target_creature: 'Creature', attack_index: int, dice_roller: Callable[[str], int], advantage_disadvantage: Optional[str] = None) -> Tuple[bool, int]:
        """ Returns (hit_status, damage_dealt) """
        if not self.can_act() or not target_creature.can_act():
            return False, 0 # (hit_status, damage_dealt)

        if not (0 <= attack_index < len(self.attacks)):
            raise IndexError(f"attack_index {attack_index} out of bounds for {self.name}'s attacks.")
            
        selected_attack = self.attacks[attack_index]

        d20_roll_str = "1d20"
        if advantage_disadvantage == 'advantage':
            d20_roll_str = "2d20kh1"
            # Potentially log: f"{self.name} attacking {target_creature.name} with advantage"
        elif advantage_disadvantage == 'disadvantage':
            d20_roll_str = "2d20kl1"
            # Potentially log: f"{self.name} attacking {target_creature.name} with disadvantage"
        # else: advantage_disadvantage is None or any other string, roll normally.
            # Potentially log: f"{self.name} attacking {target_creature.name} normally"

        d20_roll = dice_roller(d20_roll_str)
        attack_roll_result = d20_roll + selected_attack["to_hit"]
        # Add logging for the final attack_roll_result vs target_creature.ac if desired.
        
        damage_dealt = 0
        hit_status = False
        if attack_roll_result >= target_creature.ac:
            hit_status = True
            damage_dealt = dice_roller(selected_attack["damage_dice"])
            if damage_dealt < 0: 
                damage_dealt = 0 # Damage cannot be negative
            target_creature.take_damage(damage_dealt)
        
        return hit_status, damage_dealt

    def move(self, dx: int, dy: int, map_width: int, map_height: int) -> bool:
        if not self.can_act(): return False
        cost_of_movement = abs(dx) + abs(dy)
        if self.speed_remaining >= cost_of_movement:
            new_x = self.position[0] + dx
            new_y = self.position[1] + dy
            if 0 <= new_x < map_width and 0 <= new_y < map_height:
                self.position[0] = new_x
                self.position[1] = new_y
                self.speed_remaining -= cost_of_movement
                return True
        return False

    def can_act(self) -> bool:
        return self.is_alive

    def reset(self, initial_position: List[int], max_hp: Optional[int] = None, speed_total: Optional[int] = None) -> None:
        if max_hp is not None: self.max_hp = max_hp
        self.current_hp = self.max_hp
        
        if len(initial_position) != 2:
            raise ValueError("initial_position must be a list of two integers [x, y] for reset.")
        self.position = list(initial_position)
        
        if speed_total is not None: self.speed_total = speed_total
        self.speed_remaining = self.speed_total # Reset speed at the start of a turn/reset
        
        self.is_alive = self.current_hp > 0

        # Reset combat state attributes
        self.used_action_this_turn = False
        self.used_bonus_action_this_turn = False
        self.used_reaction_this_round = False
        self.is_dodging = False
        self.is_disengaging = False
        self.took_dash_action_this_turn = False

    def start_new_turn(self) -> None:
        # Dodging effect lasts until the start of this turn, so reset it now.
        self.is_dodging = False
        # Disengage effect also typically ends at the start of the next turn or after movement.
        self.is_disengaging = False # Or handle this more granularly if disengage ends after any move.

        # Reset speed. Dash effect from previous turn ends.
        self.speed_remaining = self.speed_total # Base speed for the new turn
        self.took_dash_action_this_turn = False # Dash action itself is for the current turn

        self.used_action_this_turn = False
        self.used_bonus_action_this_turn = False
        # self.used_reaction_this_round = False # Reaction resets at start of ROUND, not turn. Keep as is.
                                            # Task says reset reaction here, so I will.
        self.used_reaction_this_round = False


    def __repr__(self) -> str:
        return (f"Creature(name='{self.name}', ac={self.ac}, hp={self.current_hp}/{self.max_hp}, "
                f"pos={self.position}, speed_rem={self.speed_remaining}, alive={self.is_alive})")


class DnDCombatEnv(gym.Env):
    metadata = {'render_modes': ['ansi', 'human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, 
                 map_width: int, 
                 map_height: int, 
                 agent_stats: dict, 
                 enemy_stats: dict, 
                 grid_size: int = 1, 
                 render_mode: Optional[str] = None, 
                 export_frames_path: Optional[str] = None,
                 enemy_model_path: Optional[str] = None): # New parameter
        super().__init__()

        self.map_width = map_width
        self.map_height = map_height
        self.grid_size = grid_size 
        
        self.max_episode_steps = 100
        self.current_episode_steps = 0
        self.current_episode_num = 0
        self.agent_micro_steps_this_turn = 0

        self.render_mode = render_mode
        self.export_frames_path = export_frames_path
        
        if self.export_frames_path:
            os.makedirs(self.export_frames_path, exist_ok=True)
            
        self.screen = None
        self.clock = None
        self.font = None
        self.render_surface = None # For rgb_array
        self.window_surface = None # Surface to draw on (screen or render_surface)
        
        self.CELL_SIZE = 50
        self.STATS_AREA_HEIGHT = 100
        self.TEXT_COLOR = (200, 200, 200) # Light grey for text
        self.AGENT_COLOR = (0, 128, 0)   # Dark Green
        self.ENEMY_COLOR = (128, 0, 0)   # Dark Red
        self.GRID_COLOR = (50, 50, 50)   # Dark Grey
        self.BACKGROUND_COLOR = (0, 0, 0) # Black

        if self.render_mode in ['human', 'rgb_array']:
            if pygame is None:
                raise ImportError("Pygame is not installed, but is required for 'human' or 'rgb_array' render modes. Please install it (e.g., pip install pygame).")
            
            pygame.font.init() # Initialize font module explicitly
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18) # For grid coordinates or smaller text

            screen_width = self.map_width * self.CELL_SIZE
            screen_height = self.map_height * self.CELL_SIZE + self.STATS_AREA_HEIGHT

            if self.render_mode == 'human':
                pygame.init() # Initialize all Pygame modules
                pygame.display.set_caption("D&D Combat Environment")
                self.screen = pygame.display.set_mode((screen_width, screen_height))
                self.clock = pygame.time.Clock()
                self.window_surface = self.screen
            elif self.render_mode == 'rgb_array':
                self.render_surface = pygame.Surface((screen_width, screen_height))
                self.window_surface = self.render_surface


        # Ensure 'initial_position' is not passed directly if it's part of agent_stats to avoid TypeError
        # Store stat templates BEFORE creating dummy env for flattener
        self.agent_stats_template = agent_stats.copy()
        self.enemy_stats_template = enemy_stats.copy()
        
        # Enemy model attributes
        self.enemy_model_path = enemy_model_path
        self.enemy_model = None
        self.enemy_is_model_controlled = False
        self.enemy_obs_flattener = None

        if self.enemy_model_path and SB3_AVAILABLE:
            if os.path.exists(self.enemy_model_path):
                print(f"Attempting to load enemy model from: {self.enemy_model_path}")
                try:
                    # Important: Pass device='auto' or 'cpu' if running in CPU-only environment
                    self.enemy_model = DQN.load(self.enemy_model_path, device='auto') 
                    self.enemy_is_model_controlled = True
                    
                    print("Creating dummy environment for enemy observation flattener...")
                    # Enemy sees itself as 'agent' and current agent as 'enemy'
                    # This dummy env is only for observation flattening structure, not for running steps.
                    dummy_env_for_flattener = DnDCombatEnv(
                        map_width=self.map_width, 
                        map_height=self.map_height, 
                        agent_stats=self.enemy_stats_template,  # Enemy is the 'agent' in its own model
                        enemy_stats=self.agent_stats_template,  # Main agent is the 'enemy' from enemy model's POV
                        grid_size=self.grid_size,
                        render_mode=None, # No rendering for this dummy env
                        export_frames_path=None, # No frame export
                        enemy_model_path=None # CRITICAL: No recursive model loading
                    )
                    self.enemy_obs_flattener = FlattenObservation(dummy_env_for_flattener)
                    print("Enemy model loaded and flattener created successfully.")
                except Exception as e:
                    print(f"Error loading enemy model from {self.enemy_model_path}: {e}. Enemy will use rule-based AI.")
                    self.enemy_model = None 
                    self.enemy_is_model_controlled = False
            else:
                print(f"Warning: Enemy model path not found: {self.enemy_model_path}. Enemy will use rule-based AI.")
        elif self.enemy_model_path and not SB3_AVAILABLE:
            print("Warning: Stable Baselines3 not available. Cannot load enemy model. Enemy will use rule-based AI.")

        # The actual initial position will be set in reset()
        # Use the stored templates for creating the actual agent and enemy instances
        agent_init_stats = self.agent_stats_template.copy()
        agent_init_stats.pop('initial_position', None) 
        self.agent = Creature(name="agent", initial_position=[0,0], **agent_init_stats)

        enemy_init_stats = self.enemy_stats_template.copy()
        enemy_init_stats.pop('initial_position', None)
        self.enemy = Creature(name="enemy", initial_position=[0,0], **enemy_init_stats)
        
        self.dice_roller = roll

        # Build action maps
        self._agent_action_map = self._build_action_map_for_creature(self.agent)
        self.action_space = spaces.Discrete(len(self._agent_action_map))

        self._enemy_action_map_for_validation = None
        if self.enemy_is_model_controlled: # This flag is set after attempting to load enemy_model
            self._enemy_action_map_for_validation = self._build_action_map_for_creature(self.enemy)
            if self._enemy_action_map_for_validation is None:
                 print("Warning: Could not build enemy action map even if model controlled.")
        
        # Define Observation Space
        max_coord = max(self.map_width, self.map_height) -1
        max_dist = self.map_width + self.map_height # Max Manhattan distance
        
        self.observation_space = spaces.Dict({
            "agent_hp_norm": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "enemy_hp_norm": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "agent_pos": spaces.Box(low=0, high=max_coord, shape=(2,), dtype=np.int32),
            "enemy_pos": spaces.Box(low=0, high=max_coord, shape=(2,), dtype=np.int32),
            "distance_to_enemy": spaces.Box(low=0, high=max_dist, shape=(1,), dtype=np.float32),
            "agent_speed_remaining_norm": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "agent_used_action": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "agent_used_bonus_action": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "agent_used_reaction": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "agent_is_dodging": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })
        
        # self.render_mode = 'ansi' # Default, can be set by user # render_mode is now an __init__ param

    def _get_obs(self) -> Dict[str, Any]:
        agent_hp_norm = np.array([self.agent.current_hp / self.agent.max_hp if self.agent.max_hp > 0 else 0.0], dtype=np.float32)
        enemy_hp_norm = np.array([self.enemy.current_hp / self.enemy.max_hp if self.enemy.max_hp > 0 else 0.0], dtype=np.float32)
        
        dist = self._calculate_manhattan_distance(self.agent.position, self.enemy.position)
        
        # Clamp speed normalization to ensure it doesn't exceed 1.0 due to bonuses
        raw_speed_norm = self.agent.speed_remaining / self.agent.speed_total if self.agent.speed_total > 0 else 0.0
        agent_speed_norm = np.array([min(1.0, raw_speed_norm)], dtype=np.float32)


        return {
            "agent_hp_norm": agent_hp_norm,
            "enemy_hp_norm": enemy_hp_norm,
            "agent_pos": np.array(self.agent.position, dtype=np.int32),
            "enemy_pos": np.array(self.enemy.position, dtype=np.int32),
            "distance_to_enemy": np.array([dist], dtype=np.float32),
            "agent_speed_remaining_norm": agent_speed_norm,
            "agent_used_action": np.array([1.0 if self.agent.used_action_this_turn else 0.0], dtype=np.float32),
            "agent_used_bonus_action": np.array([1.0 if self.agent.used_bonus_action_this_turn else 0.0], dtype=np.float32),
            "agent_used_reaction": np.array([1.0 if self.agent.used_reaction_this_round else 0.0], dtype=np.float32),
            "agent_is_dodging": np.array([1.0 if self.agent.is_dodging else 0.0], dtype=np.float32),
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "agent_hp": self.agent.current_hp,
            "enemy_hp": self.enemy.current_hp,
            "agent_max_hp": self.agent.max_hp,
            "enemy_max_hp": self.enemy.max_hp,
            "distance": self._calculate_manhattan_distance(self.agent.position, self.enemy.position)
        }

    def _get_enemy_observation(self) -> Dict[str, Any]:
        # From enemy's perspective:
        # 'agent' is the enemy itself, 'enemy' is the current agent.

        # Normalize HP
        # Enemy's HP (becomes 'agent_hp_norm' for its model)
        enemy_self_hp_norm = self.enemy.current_hp / self.enemy.max_hp if self.enemy.max_hp > 0 else 0.0
        # Current agent's HP (becomes 'enemy_hp_norm' for enemy's model)
        current_agent_hp_norm = self.agent.current_hp / self.agent.max_hp if self.agent.max_hp > 0 else 0.0
        
        # Normalize speed for the enemy (becomes 'agent_speed_remaining_norm')
        # Ensure speed_total is not zero to avoid division by zero, and clamp at 1.0
        raw_enemy_speed_norm = self.enemy.speed_remaining / self.enemy.speed_total if self.enemy.speed_total > 0 else 0.0
        enemy_self_speed_norm = min(1.0, raw_enemy_speed_norm)


        obs = {
            # Perspective shift:
            "agent_hp_norm": np.array([enemy_self_hp_norm], dtype=np.float32),
            "enemy_hp_norm": np.array([current_agent_hp_norm], dtype=np.float32),
            
            "agent_pos": np.array(self.enemy.position, dtype=np.int32), # Enemy's own position
            "enemy_pos": np.array(self.agent.position, dtype=np.int32),   # Current agent's position
            
            # Distance is symmetrical
            "distance_to_enemy": np.array([self._calculate_manhattan_distance(self.enemy.position, self.agent.position)], dtype=np.float32),
            
            # Enemy's speed
            "agent_speed_remaining_norm": np.array([enemy_self_speed_norm], dtype=np.float32),
            # Enemy's action economy states (from its perspective, so "agent_...")
            "agent_used_action": np.array([1.0 if self.enemy.used_action_this_turn else 0.0], dtype=np.float32),
            "agent_used_bonus_action": np.array([1.0 if self.enemy.used_bonus_action_this_turn else 0.0], dtype=np.float32),
            "agent_used_reaction": np.array([1.0 if self.enemy.used_reaction_this_round else 0.0], dtype=np.float32),
            "agent_is_dodging": np.array([1.0 if self.enemy.is_dodging else 0.0], dtype=np.float32),
        }
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)

        # Randomly place agent and enemy, ensuring no overlap
        while True:
            agent_start_pos = [self.np_random.integers(0, self.map_width), self.np_random.integers(0, self.map_height)]
            enemy_start_pos = [self.np_random.integers(0, self.map_width), self.np_random.integers(0, self.map_height)]
            if agent_start_pos != enemy_start_pos:
                break
        
        # Reset creatures (max_hp and speed_total are taken from their initial stats)
        self.agent.reset(initial_position=agent_start_pos)
        self.enemy.reset(initial_position=enemy_start_pos)
        
        self.current_episode_steps = 0 # Reset step counter for the new episode
        self.current_episode_num += 1 # Increment episode number
        self.agent_micro_steps_this_turn = 0

        return self._get_obs(), self._get_info()

    def _calculate_manhattan_distance(self, pos1: List[int], pos2: List[int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_valid_position(self, x: int, y: int) -> bool:
        return 0 <= x < self.map_width and 0 <= y < self.map_height

    @property # Make it accessible like a variable but defined as a method
    def action_map(self):
        # This property is what external things (like tests or example runners) might use.
        # It should reflect the agent's perspective.
        return self._agent_action_map

    def _build_action_map_for_creature(self, creature: Creature) -> List[Dict[str, Any]]:
        action_map = []
        # 1. Move actions (NSEW, 1 cell at a time) - These are standard
        action_map.append({"type": "move", "delta": (0, -1), "name": "move_N"}) # North
        action_map.append({"type": "move", "delta": (0, 1), "name": "move_S"})  # South
        action_map.append({"type": "move", "delta": (1, 0), "name": "move_E"})  # East
        action_map.append({"type": "move", "delta": (-1, 0), "name": "move_W"}) # West

        # 2. Attack actions - Specific to the creature
        for i, attack_details in enumerate(creature.attacks):
            action_map.append({"type": "attack", "index": i, "name": f"attack_{attack_details['name']}"})

        # 3. Bonus Actions - Specific to the creature
        for i, ba_name_or_dict in enumerate(creature.bonus_actions):
            if isinstance(ba_name_or_dict, str):
                action_map.append({"type": "bonus_action", "name": ba_name_or_dict, "index": i})
            elif isinstance(ba_name_or_dict, dict): # If bonus actions become more structured
                 action_map.append({
                     "type": "bonus_action",
                     "name": ba_name_or_dict.get("name", f"bonus_action_{i}"),
                     "index": i,
                     **ba_name_or_dict # Merge other details if any
                 })

        # 4. Standard D&D Actions (Dash, Disengage, Dodge)
        action_map.append({"type": "dash", "name": "dash"})
        action_map.append({"type": "disengage", "name": "disengage"})
        action_map.append({"type": "dodge", "name": "dodge"})

        # 5. End Turn action (hands the round over to the opponent)
        action_map.append({"type": "end_turn", "name": "end_turn"})

        return action_map

    def _decode_action(self, action_int: int, is_enemy: bool = False) -> Dict[str, Any]:
        map_to_use = self._agent_action_map # Default to agent's map
        actor_for_map_construction = self.agent # For debug message

        if is_enemy and self.enemy_is_model_controlled and self._enemy_action_map_for_validation is not None:
            map_to_use = self._enemy_action_map_for_validation
            actor_for_map_construction = self.enemy # For debug message
            # print(f"DEBUG: Decoding enemy action using enemy_action_map. Action int: {action_int}, Map size: {len(map_to_use)}")
        # else:
            # print(f"DEBUG: Decoding agent action using agent_action_map. Action int: {action_int}, Map size: {len(map_to_use)}")

        if not (0 <= action_int < len(map_to_use)):
            if is_enemy:
                # An enemy model trained against a different action space may
                # predict an id we don't have; treat it as a pass instead of
                # crashing the episode.
                return {"type": "pass", "name": f"pass_invalid_action_id_{action_int}"}
            actor_type = "enemy" if is_enemy else "agent"
            map_type_used = "enemy_specific_map" if (is_enemy and self.enemy_is_model_controlled and self._enemy_action_map_for_validation is not None) else "agent_map"

            err_msg = (
                f"Invalid action integer: {action_int} for actor '{actor_type}'. "
                f"Attempted to use {map_type_used} with size {len(map_to_use)}. "
                f"Actor's actual attacks: {len(self.enemy.attacks if is_enemy else self.agent.attacks)}, "
                f"Actor's actual bonus_actions: {len(self.enemy.bonus_actions if is_enemy else self.agent.bonus_actions)}."
            )
            move_actions_in_map = sum(1 for action in map_to_use if action["type"] == "move")
            attack_actions_in_map = sum(1 for action in map_to_use if action["type"] == "attack")
            bonus_actions_in_map = sum(1 for action in map_to_use if action["type"] == "bonus_action")
            standard_actions_in_map = sum(1 for action in map_to_use if action["type"] in ["dash", "disengage", "dodge"])
            pass_end_actions_in_map = sum(1 for action in map_to_use if action["type"] in ["pass_turn", "end_turn"])

            err_msg += (
                f" Map structure (used for decoding): Moves={move_actions_in_map}, Attacks={attack_actions_in_map}, "
                f"BonusActions={bonus_actions_in_map}, StandardActions={standard_actions_in_map}, "
                f"Pass/End={pass_end_actions_in_map}."
            )
            raise ValueError(err_msg)

        action_details = map_to_use[action_int].copy()

        actor = self.enemy if is_enemy else self.agent

        if action_details["type"] == "attack":
            attack_idx = action_details.get("index")
            if attack_idx is None or not (0 <= attack_idx < len(actor.attacks)):
                # Fallback if model chooses an attack index that the current actor doesn't have
                return {"type": "pass", "name": f"pass_invalid_attack_idx_{attack_idx}"}
            # Update name for clarity, index is used directly on actor's attack list
            action_details["name"] = f"attack_{actor.attacks[attack_idx].get('name', f'idx{attack_idx}')}"
        
        elif action_details["type"] == "bonus_action":
            bonus_idx = action_details.get("index")
            if bonus_idx is None or not (0 <= bonus_idx < len(actor.bonus_actions)):
                # Fallback for invalid bonus action index
                return {"type": "pass", "name": f"pass_invalid_bonus_idx_{bonus_idx}"}
            # Update name for clarity
            action_details["name"] = f"bonus_{actor.bonus_actions[bonus_idx]}" # Assumes bonus_actions are strings (names)
            # If bonus_actions were dicts, it would be: actor.bonus_actions[bonus_idx].get('name', f'idx{bonus_idx}')
        
        return action_details

    def _first_melee_attack_index(self, creature: Creature) -> int:
        for i, attack in enumerate(creature.attacks):
            if attack.get("attack_type", "melee") == "melee":
                return i
        return -1

    def _turn_economy_exhausted(self, creature: Creature) -> bool:
        """True when the creature has nothing meaningful left to do this turn."""
        bonus_done = creature.used_bonus_action_this_turn or not creature.bonus_actions
        return creature.used_action_this_turn and bonus_done and creature.speed_remaining <= 0

    def _max_micro_steps(self, creature: Creature) -> int:
        # Enough for a full Dash turn (2x speed in single-cell moves) plus an
        # action, a bonus action, and some slack. Bounds turn length so a
        # policy repeating invalid actions cannot stall the round forever.
        return 2 * creature.speed_total + 3

    def _execute_micro_action(self, actor: Creature, target: Creature,
                              decoded_action: Dict[str, Any], info: Dict[str, Any],
                              prefix: str) -> Tuple[float, bool, bool]:
        """Execute a single micro-action for `actor` against `target`.

        Shared by the agent and the (model- or rule-controlled) enemy so both
        sides play by identical rules. Rewards are always from the agent's
        perspective. Returns (reward_delta, terminated, action_succeeded).
        """
        is_agent = actor is self.agent
        other_prefix = "enemy" if is_agent else "agent"
        reward_delta = 0.0
        terminated = False
        success = False
        action_type = decoded_action["type"]

        if action_type == "move":
            dx, dy = decoded_action["delta"]
            cost = abs(dx) + abs(dy)
            if actor.speed_remaining < cost:
                info[f'{prefix}_move_outcome'] = "Not enough speed."
            else:
                target_x = actor.position[0] + dx
                target_y = actor.position[1] + dy
                if not self._is_valid_position(target_x, target_y):
                    info[f'{prefix}_move_outcome'] = "Move out of bounds."
                elif [target_x, target_y] == target.position and target.is_alive:
                    info[f'{prefix}_move_outcome'] = f"Blocked by {other_prefix}."
                else:
                    # The move is definitely happening; resolve an opportunity
                    # attack first if the actor is leaving the target's reach.
                    if target.can_act() and \
                       self._calculate_manhattan_distance(actor.position, target.position) == 1 and \
                       self._calculate_manhattan_distance([target_x, target_y], target.position) > 1 and \
                       not actor.is_disengaging and \
                       not target.used_reaction_this_round:
                        melee_idx = self._first_melee_attack_index(target)
                        if melee_idx != -1:
                            info[f'opportunity_attack_by_{other_prefix}'] = (
                                f"{other_prefix.capitalize()} opportunity attack from {target.position} "
                                f"as {prefix} moves from {actor.position} to {[target_x, target_y]}")
                            hp_before = actor.current_hp
                            hit, _ = target.make_attack(actor, melee_idx, self.dice_roller,
                                                        advantage_disadvantage=None)
                            target.used_reaction_this_round = True
                            damage = hp_before - actor.current_hp
                            if hit:
                                reward_delta += -damage if is_agent else damage
                                info[f'opportunity_attack_by_{other_prefix}_outcome'] = f"Hit, dealt {damage} damage."
                            else:
                                info[f'opportunity_attack_by_{other_prefix}_outcome'] = "Miss."
                            if not actor.is_alive:
                                terminated = True
                                reward_delta += -100 if is_agent else 100
                                info['combat_outcome'] = ("Agent died to Opportunity Attack" if is_agent
                                                          else "Enemy died to Opportunity Attack")
                                return reward_delta, terminated, False
                    if actor.move(dx, dy, self.map_width, self.map_height):
                        success = True
                        info[f'{prefix}_move_outcome'] = f"Moved ({dx},{dy})."
                    else:
                        info[f'{prefix}_move_outcome'] = "Move failed unexpectedly."

        elif action_type == "attack":
            if actor.used_action_this_turn:
                info[f'{prefix}_action_details'] = f"{prefix.capitalize()} tried Attack but action already used."
                info[f'{prefix}_attack_outcome'] = "Action already used."
            else:
                attack_idx = decoded_action["index"]
                attack_stats = actor.attacks[attack_idx]
                attack_name = attack_stats.get("name", f"Attack {attack_idx}")
                attack_range = attack_stats.get("range", 1)
                attack_kind = attack_stats.get("attack_type", "melee")
                distance = self._calculate_manhattan_distance(actor.position, target.position)

                adv_disadv = None
                if attack_kind == "ranged" and distance <= 1:
                    adv_disadv = 'disadvantage'
                    info[f'{prefix}_attack_modifier'] = f"Ranged attack at disadvantage ({other_prefix} adjacent)."
                if target.is_dodging:
                    if adv_disadv == 'advantage':
                        adv_disadv = None
                        info[f'{prefix}_attack_modifier'] = (info.get(f'{prefix}_attack_modifier', "") +
                                                             f" {other_prefix.capitalize()} dodging cancels advantage.")
                    else:
                        adv_disadv = 'disadvantage'
                        info[f'{prefix}_attack_modifier'] = (info.get(f'{prefix}_attack_modifier', "") +
                                                             f" {other_prefix.capitalize()} dodging, attack at disadvantage.")

                info[f'{prefix}_action_details'] = (
                    f"Attack: {attack_name} (Range: {attack_range}, Type: {attack_kind}, "
                    f"Adv/Disadv: {adv_disadv}), Target Dist: {distance}")

                if distance <= attack_range:
                    hp_before = target.current_hp
                    hit, _ = actor.make_attack(target, attack_idx, self.dice_roller,
                                               advantage_disadvantage=adv_disadv)
                    damage = hp_before - target.current_hp
                    if hit:
                        reward_delta += damage if is_agent else -damage
                        info[f'{prefix}_attack_outcome'] = f"Hit, dealt {damage} damage."
                    else:
                        info[f'{prefix}_attack_outcome'] = "Miss."
                    if not target.is_alive:
                        terminated = True
                        reward_delta += 100 if is_agent else -100
                        info['combat_outcome'] = "Agent won" if is_agent else "Enemy won"
                else:
                    info[f'{prefix}_attack_outcome'] = "Out of range."
                actor.used_action_this_turn = True
                success = True

        elif action_type == "bonus_action":
            if actor.used_bonus_action_this_turn:
                info[f'{prefix}_action_details'] = f"{prefix.capitalize()} tried Bonus Action but it was already used."
                info[f'{prefix}_bonus_outcome'] = "Bonus action already used."
            else:
                ba_name = decoded_action["name"]
                info[f'{prefix}_action_details'] = f"Bonus Action: {ba_name}"
                if ba_name == "bonus_move_1_cell":
                    actor.speed_remaining += 1
                    if is_agent:
                        reward_delta += 0.02
                    info[f'{prefix}_bonus_outcome'] = "Used bonus_move_1_cell, gained 1 speed."
                else:
                    info[f'{prefix}_bonus_outcome'] = f"Used bonus action: {ba_name} (effect placeholder)."
                actor.used_bonus_action_this_turn = True
                success = True

        elif action_type in ("dash", "disengage", "dodge"):
            if actor.used_action_this_turn:
                info[f'{prefix}_action_details'] = (f"{prefix.capitalize()} tried {action_type.capitalize()} "
                                                    f"but action already used.")
            else:
                if action_type == "dash":
                    actor.took_dash_action_this_turn = True
                    actor.speed_remaining += actor.speed_total
                elif action_type == "disengage":
                    actor.is_disengaging = True
                else:
                    actor.is_dodging = True
                actor.used_action_this_turn = True
                success = True
                if is_agent:
                    reward_delta += 0.05
                info[f'{prefix}_action_details'] = f"{prefix.capitalize()} used {action_type.capitalize()} action."

        elif action_type == "end_turn":
            success = True
            info[f'{prefix}_action_details'] = f"{prefix.capitalize()} ended turn."

        elif action_type == "pass":
            # Fallback from _decode_action for an index this creature doesn't have.
            info[f'{prefix}_action_details'] = (f"{prefix.capitalize()} chose an action invalid for this "
                                                f"creature; treated as pass.")

        return reward_delta, terminated, success

    def _run_enemy_turn(self, info: Dict[str, Any]) -> Tuple[float, bool]:
        """Run the enemy's complete turn. Returns (reward_delta, terminated)."""
        self.enemy.start_new_turn()
        if self.enemy_is_model_controlled and self.enemy_model and self.enemy_obs_flattener:
            return self._run_enemy_model_turn(info)
        return self._run_enemy_rule_based_turn(info)

    def _run_enemy_model_turn(self, info: Dict[str, Any]) -> Tuple[float, bool]:
        info['enemy_ai_type'] = 'model_controlled'
        reward_delta = 0.0
        terminated = False
        actions_taken = []
        for _ in range(self._max_micro_steps(self.enemy)):
            enemy_obs_dict = self._get_enemy_observation()
            flat_enemy_obs = self.enemy_obs_flattener.observation(enemy_obs_dict)
            enemy_action_id_array, _ = self.enemy_model.predict(flat_enemy_obs, deterministic=True)
            enemy_action_id = int(enemy_action_id_array.item())
            decoded = self._decode_action(enemy_action_id, is_enemy=True)
            actions_taken.append(decoded.get("name", str(enemy_action_id)))

            r, terminated, success = self._execute_micro_action(self.enemy, self.agent, decoded, info, 'enemy')
            reward_delta += r
            if terminated or decoded["type"] == "end_turn" or self._turn_economy_exhausted(self.enemy):
                break
            if not success:
                # A failed action leaves the observation unchanged, so a
                # deterministic policy would repeat it forever; end the turn.
                break
        info['enemy_action_choice'] = ", ".join(actions_taken)
        return reward_delta, terminated

    def _run_enemy_rule_based_turn(self, info: Dict[str, Any]) -> Tuple[float, bool]:
        info['enemy_ai_type'] = 'rule_based'
        reward_delta = 0.0
        terminated = False
        attack_stats = self.enemy.attacks[0] if self.enemy.attacks else None
        attack_range = attack_stats.get("range", 1) if attack_stats else 1

        # Close distance until within attack range (or adjacent if unarmed).
        while (not terminated and self.enemy.speed_remaining > 0 and
               self._calculate_manhattan_distance(self.enemy.position, self.agent.position) > attack_range):
            diff_x = self.agent.position[0] - self.enemy.position[0]
            diff_y = self.agent.position[1] - self.enemy.position[1]
            step_x = (1 if diff_x > 0 else -1, 0) if diff_x != 0 else None
            step_y = (0, 1 if diff_y > 0 else -1) if diff_y != 0 else None
            ordered = (step_x, step_y) if abs(diff_x) >= abs(diff_y) else (step_y, step_x)
            candidate_steps = [s for s in ordered if s is not None]

            moved = False
            for delta in candidate_steps:
                decoded = {"type": "move", "delta": delta, "name": f"move({delta[0]},{delta[1]})"}
                r, terminated, success = self._execute_micro_action(self.enemy, self.agent, decoded, info, 'enemy')
                reward_delta += r
                if terminated:
                    return reward_delta, terminated
                if success:
                    moved = True
                    break
            if not moved:
                break  # Boxed in; stop trying to move.

        if (not terminated and attack_stats and not self.enemy.used_action_this_turn and
                self._calculate_manhattan_distance(self.enemy.position, self.agent.position) <= attack_range):
            decoded = {"type": "attack", "index": 0,
                       "name": f"attack_{attack_stats.get('name', '0')}"}
            r, terminated, _ = self._execute_micro_action(self.enemy, self.agent, decoded, info, 'enemy')
            reward_delta += r

        return reward_delta, terminated

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self.current_episode_steps += 1
        info: Dict[str, Any] = {}
        terminated = False
        truncated = False
        reward = -0.1  # Flat per-step penalty

        decoded_action = self._decode_action(action)
        info['agent_action'] = decoded_action

        # Agent micro-action. The agent keeps acting across successive step()
        # calls until it ends its turn (explicitly, by exhausting its action
        # economy, or by hitting the per-turn micro-step cap); only then does
        # the enemy take its full turn.
        agent_turn_over = True
        if self.agent.can_act():
            reward_delta, terminated, success = self._execute_micro_action(
                self.agent, self.enemy, decoded_action, info, 'agent')
            reward += reward_delta
            if not success and not terminated and decoded_action["type"] != "end_turn":
                reward -= 0.05  # Chose an unavailable or invalid action
            self.agent_micro_steps_this_turn += 1
            agent_turn_over = (
                decoded_action["type"] == "end_turn"
                or self._turn_economy_exhausted(self.agent)
                or self.agent_micro_steps_this_turn >= self._max_micro_steps(self.agent)
            )

        # Enemy's full turn, once the agent's turn is over.
        if not terminated and agent_turn_over and self.enemy.can_act():
            reward_delta, terminated = self._run_enemy_turn(info)
            reward += reward_delta

        # Refresh the agent for its next turn so the returned observation
        # describes the state it will actually decide from.
        if not terminated and agent_turn_over and self.agent.can_act():
            self.agent.start_new_turn()
            self.agent_micro_steps_this_turn = 0

        if not terminated and self.current_episode_steps >= self.max_episode_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {**self._get_info(), **info}

    def _render_pygame_frame(self, surface: pygame.Surface):
        if surface is None: return

        surface.fill(self.BACKGROUND_COLOR)

        # Draw Grid
        for x in range(self.map_width):
            for y in range(self.map_height):
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(surface, self.GRID_COLOR, rect, 1) # Grid lines
                # Optional: Draw grid coordinates
                # coord_text = self.small_font.render(f"{x},{y}", True, self.GRID_COLOR)
                # surface.blit(coord_text, (rect.x + 2, rect.y + 2))


        # Draw Creatures
        if self.agent.is_alive:
            agent_rect = pygame.Rect(self.agent.position[0] * self.CELL_SIZE, 
                                     self.agent.position[1] * self.CELL_SIZE, 
                                     self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(surface, self.AGENT_COLOR, agent_rect)
            agent_id_text = self.small_font.render("A", True, self.TEXT_COLOR)
            surface.blit(agent_id_text, (agent_rect.x + self.CELL_SIZE // 2 - agent_id_text.get_width() // 2, 
                                       agent_rect.y + self.CELL_SIZE // 2 - agent_id_text.get_height() // 2))


        if self.enemy.is_alive:
            enemy_rect = pygame.Rect(self.enemy.position[0] * self.CELL_SIZE, 
                                     self.enemy.position[1] * self.CELL_SIZE, 
                                     self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(surface, self.ENEMY_COLOR, enemy_rect)
            enemy_id_text = self.small_font.render("E", True, self.TEXT_COLOR)
            surface.blit(enemy_id_text, (enemy_rect.x + self.CELL_SIZE // 2 - enemy_id_text.get_width() // 2,
                                        enemy_rect.y + self.CELL_SIZE // 2 - enemy_id_text.get_height() // 2))

        # Draw Stats Area Background
        stats_area_y_start = self.map_height * self.CELL_SIZE
        stats_rect = pygame.Rect(0, stats_area_y_start, 
                                 self.map_width * self.CELL_SIZE, self.STATS_AREA_HEIGHT)
        pygame.draw.rect(surface, (30,30,30), stats_rect) # Slightly lighter than background for stats area

        # Draw Stats Text
        agent_stats_str = (f"Agent: HP {self.agent.current_hp}/{self.agent.max_hp} | "
                           f"AC {self.agent.ac} | Speed {self.agent.speed_remaining}/{self.agent.speed_total}")
        enemy_stats_str = (f"Enemy: HP {self.enemy.current_hp}/{self.enemy.max_hp} | "
                           f"AC {self.enemy.ac} | Speed {self.enemy.speed_remaining}/{self.enemy.speed_total}")

        agent_text_render = self.font.render(agent_stats_str, True, self.TEXT_COLOR)
        enemy_text_render = self.font.render(enemy_stats_str, True, self.TEXT_COLOR)

        surface.blit(agent_text_render, (10, stats_area_y_start + 10))
        surface.blit(enemy_text_render, (10, stats_area_y_start + 40))
        
        # Optional: Display last action or current turn info
        # last_action_info = f"Last Action: {self._last_action_string}" # If you store this
        # last_action_render = self.font.render(last_action_info, True, self.TEXT_COLOR)
        # surface.blit(last_action_render, (10, stats_area_y_start + 70))


    def render(self) -> Optional[Union[np.ndarray, str]]:
        if self.render_mode is None and 'human' not in self.metadata['render_modes'] and 'rgb_array' not in self.metadata['render_modes']:
             gym.logger.warn("Cannot render without specifying a render mode and Pygame installed. Using 'ansi'.")
             # Fallback to ANSI if pygame is not available or modes are not set up.
             # However, if pygame is None from the start, __init__ would have raised error if human/rgb was selected.
             self.render_mode = 'ansi' # Default if truly no other option

        if self.render_mode == 'human':
            if pygame is None or self.screen is None or self.clock is None:
                # This case should ideally be prevented by __init__ erroring out
                # or by not allowing 'human' mode if pygame is not available.
                print("Pygame not initialized for 'human' mode. Cannot render.")
                return None 
            
            self._render_pygame_frame(self.screen)

            if self.export_frames_path:
                step_num_for_filename = self.current_episode_steps if hasattr(self, 'current_episode_steps') else 0
                episode_num_for_filename = self.current_episode_num if hasattr(self, 'current_episode_num') else 0
                
                filename = os.path.join(self.export_frames_path, f"e{episode_num_for_filename:03d}_s{step_num_for_filename:04d}.png")
                try:
                    pygame.image.save(self.screen, filename)
                except Exception as e:
                    print(f"Error saving frame {filename}: {e}")

            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close() # This will call pygame.quit()
                    # To signal the main loop to stop (if run_env.py is the main loop)
                    # this environment might need a flag like self.running = False
                    # For now, close() handles pygame shutdown. The outer loop should check termination.
            return None # Human mode usually returns None

        elif self.render_mode == 'rgb_array':
            if pygame is None or self.render_surface is None:
                print("Pygame not initialized for 'rgb_array' mode. Cannot render.")
                return None
            
            self._render_pygame_frame(self.render_surface)
            # Convert surface to numpy array and transpose for (H, W, C)
            return np.transpose(pygame.surfarray.array3d(self.render_surface), axes=(1, 0, 2))

        elif self.render_mode == 'ansi':
            grid = [['.' for _ in range(self.map_width)] for _ in range(self.map_height)]
            
            if self.enemy.is_alive and self._is_valid_position(self.enemy.position[0], self.enemy.position[1]):
                grid[self.enemy.position[1]][self.enemy.position[0]] = 'E'
            
            if self.agent.is_alive and self._is_valid_position(self.agent.position[0], self.agent.position[1]):
                grid[self.agent.position[1]][self.agent.position[0]] = 'A'

            ansi_str = "\n".join(" ".join(row) for row in grid)
            ansi_str += (f"\nAgent HP: {self.agent.current_hp}/{self.agent.max_hp}, Speed: {self.agent.speed_remaining}/{self.agent.speed_total} "
                         f"| Enemy HP: {self.enemy.current_hp}/{self.enemy.max_hp}")
            print(ansi_str) # ANSI mode prints to console
            return ansi_str # And also returns the string for gym compliance
        
        else:
            # Should not happen if render_mode is validated in __init__ or here
            raise ValueError(f"Unsupported render mode: {self.render_mode}")


    def close(self):
        if self.render_mode == 'human' and pygame is not None and self.screen is not None:
            try:
                pygame.display.quit()
                pygame.quit()
                self.screen = None # Mark as closed
            except Exception as e:
                print(f"Error during pygame shutdown: {e}")
        # No specific cleanup needed for rgb_array surface beyond normal GC
        # No specific cleanup for ansi

if __name__ == '__main__':
    # Example Usage (requires dice.py in the same directory or PYTHONPATH)
    
    agent_example_stats = {
        "ac": 15,
        "max_hp": 30,
        "speed_total": 6, # 6 cells (e.g. 30ft / 5ft per cell)
        "attacks": [
            {"name": "sword", "to_hit": 5, "damage_dice": "1d8+3", "num_attacks": 1, "attack_type": "melee", "range": 1},
            {"name": "dagger_offhand", "to_hit": 5, "damage_dice": "1d4+3", "num_attacks": 1, "attack_type": "melee", "range": 1}
        ],
        "bonus_actions": ["bonus_move_1_cell"] 
        # "initial_position" is not needed here as env sets it in reset
    }

    enemy_example_stats = {
        "ac": 13,
        "max_hp": 15,
        "speed_total": 6,
        "attacks": [
            {"name": "scimitar", "to_hit": 4, "damage_dice": "1d6+2", "num_attacks": 1, "attack_type": "melee", "range": 1}
        ],
        "bonus_actions": []
    }

    env = DnDCombatEnv(map_width=10, map_height=10, 
                       agent_stats=agent_example_stats, 
                        enemy_stats=enemy_example_stats,
                        render_mode='human') # Example with human rendering

    obs, info = env.reset()
    print("Initial Observation:", obs)
    print("Initial Info:", info)
    env.render()

    # Example: Agent tries to attack enemy (action index for first attack is 4: N,S,E,W, Attack0)
    # Find the sword attack index: (Now 'club' for commoner if using bestiary stats)
    # For the __main__ example, let's use the generic first attack.
    first_attack_action_idx = -1
    for i, action_def in enumerate(env.action_map):
        if action_def["type"] == "attack":
            first_attack_action_idx = i
            print(f"Found first attack: {action_def['name']} at index {i}")
            break
    
    if first_attack_action_idx != -1:
        print(f"\nTaking action: {env.action_map[first_attack_action_idx]['name']} (action index {first_attack_action_idx})")
        obs, reward, terminated, truncated, info = env.step(first_attack_action_idx)
        print("Observation after attack:", obs)
        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Info:", info)
        if env.render_mode == 'human': # Only render if human mode
            env.render()
            pygame.time.wait(1000) # Pause to see result
        elif env.render_mode == 'ansi':
            env.render()

    else:
        print("Could not find any attack action for example.")

    # Example: Agent tries to move East
    move_east_action_idx = -1
    for i, action_def in enumerate(env.action_map):
        if action_def["type"] == "move" and action_def["name"] == "move_E":
            move_east_action_idx = i
            break
    if move_east_action_idx != -1:
        print(f"\nTaking action: Move East (action index {move_east_action_idx})")
        obs, reward, terminated, truncated, info = env.step(move_east_action_idx)
        print("Observation after move:", obs)
        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Info:", info)
        if env.render_mode == 'human': # Only render if human mode
             env.render()
             pygame.time.wait(1000) # Pause to see result
        elif env.render_mode == 'ansi':
            env.render()
    else:
        print("Could not find move East action for example.")
    
    if env.render_mode == 'human':
        env.close() # Important to close pygame window
