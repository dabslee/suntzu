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

        # Define Action Space
        self._action_map: List[Dict[str, Any]] = []
        # 1. Move actions (NSEW, 1 cell at a time)
        self._action_map.append({"type": "move", "delta": (0, -1), "name": "move_N"}) # North
        self._action_map.append({"type": "move", "delta": (0, 1), "name": "move_S"})  # South
        self._action_map.append({"type": "move", "delta": (1, 0), "name": "move_E"})  # East
        self._action_map.append({"type": "move", "delta": (-1, 0), "name": "move_W"}) # West
        
        # 2. Attack actions
        for i, attack_details in enumerate(self.agent.attacks):
            self._action_map.append({"type": "attack", "index": i, "name": f"attack_{attack_details['name']}"})
            
        # 3. Bonus Actions
        for i, ba_name in enumerate(self.agent.bonus_actions):
            self._action_map.append({"type": "bonus_action", "name": ba_name, "index": i})

        # 4. Pass action
        self._action_map.append({"type": "pass", "name": "pass_turn"})

        # 5. Standard Actions (Dash, Disengage, Dodge)
        self._action_map.append({"type": "dash", "name": "dash"})
        self._action_map.append({"type": "disengage", "name": "disengage"})
        self._action_map.append({"type": "dodge", "name": "dodge"})

        # 6. End turn
        self._action_map.append({"type": "end_turn", "name": "end_turn"})
        
        self.action_space = spaces.Discrete(len(self.action_map))

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
        
        # Reset turn-specific states (e.g. if bonus actions were once per turn)
        # self.bonus_action_used_this_turn = False 
        self.current_episode_steps = 0 # Reset step counter for the new episode
        self.current_episode_num += 1 # Increment episode number

        return self._get_obs(), self._get_info()

    def _calculate_manhattan_distance(self, pos1: List[int], pos2: List[int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_valid_position(self, x: int, y: int) -> bool:
        return 0 <= x < self.map_width and 0 <= y < self.map_height

    @property # Make it accessible like a variable but defined as a method
    def action_map(self):
        return self._action_map

    def _decode_action(self, action_int: int, is_enemy: bool = False) -> Dict[str, Any]:
        # The global _action_map is defined based on agent's capabilities in __init__.
        # If is_enemy is True, the model for the enemy is assumed to output an action_int
        # that maps to this same structure. The 'index' for attack/bonus actions
        # will then refer to the enemy's specific list of attacks/bonus actions.

        if not (0 <= action_int < len(self._action_map)):
            # This check uses the length of the agent-defined _action_map.
            # This implies the enemy model's action space must be compatible in size.
            raise ValueError(f"Invalid action integer: {action_int} for actor {'enemy' if is_enemy else 'agent'}")

        # Get the generic action template from the agent-defined map
        action_details = self._action_map[action_int].copy() # Use a copy to avoid modifying the template

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

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self.current_episode_steps += 1 # Increment step counter
        info = {} # Initialize info dictionary

        terminated = False
        truncated = False 
        reward = -0.1  # Overall step penalty

        decoded_action = self._decode_action(action)
        info['agent_action'] = decoded_action # Store agent's chosen action type

        # Agent's Turn
        if self.agent.can_act():
            # Call start_new_turn() for the agent at the beginning of its action processing.
            # This is a simplification. In a real game, this would be called once when the agent's turn starts,
            # not before every single action it might take in a step.
            # However, given the current structure where step() processes one agent action,
            # we need to ensure its state is fresh if this is the "first" action of its turn.
            # A simple proxy: if no action/bonus action used yet, and speed is full, it's start of turn.
            # This condition might need refinement based on how turns are managed by the calling loop.
            if not self.agent.used_action_this_turn and \
               not self.agent.used_bonus_action_this_turn and \
               self.agent.speed_remaining == self.agent.speed_total:
                self.agent.start_new_turn()

            action_taken_successfully = False
            # Note: self.agent.speed_remaining is now managed by start_new_turn and Dash.
            # The old line `self.agent.speed_remaining = self.agent.speed_total` is removed.

            if decoded_action["type"] == "move":
                # Movement does not use self.agent.used_action_this_turn
                dx, dy = decoded_action["delta"]
                cost_for_this_move_action = abs(dx) + abs(dy) # Cost is based on distance moved
                
                if self.agent.speed_remaining >= cost_for_this_move_action:
                    old_pos = list(self.agent.position)
                    target_x = self.agent.position[0] + dx
                    target_y = self.agent.position[1] + dy

                    # Opportunity Attack Check (Agent Moving)
                    if self.enemy.can_act() and \
                       self._calculate_manhattan_distance(old_pos, self.enemy.position) == 1 and \
                       self._calculate_manhattan_distance([target_x, target_y], self.enemy.position) > 1 and \
                       not self.agent.is_disengaging and \
                       not self.enemy.used_reaction_this_round:

                        # Check if enemy has a melee attack
                        enemy_melee_attack_idx = -1
                        for i, attack in enumerate(self.enemy.attacks):
                            if attack.get("attack_type", "melee") == "melee": # Assume melee if not specified
                                enemy_melee_attack_idx = i
                                break

                        if enemy_melee_attack_idx != -1:
                            info['opportunity_attack_by_enemy'] = f"Enemy Opportunity Attack on Agent from {self.enemy.position} to {old_pos} as agent moves to {[target_x, target_y]}"
                            # Determine adv/disadv for OA (typically None, but consider if target is dodging - agent isn't dodging when it moves)
                            oa_adv_disadv = None # No advantage/disadvantage for standard OA

                            hit, damage_dealt = self.enemy.make_attack(self.agent, enemy_melee_attack_idx, self.dice_roller, advantage_disadvantage=oa_adv_disadv)
                            if hit:
                                reward -= damage_dealt # Negative reward for agent taking damage
                                info['opportunity_attack_by_enemy_outcome'] = f"Hit, dealt {damage_dealt} damage."
                            else:
                                info['opportunity_attack_by_enemy_outcome'] = "Miss."
                            self.enemy.used_reaction_this_round = True
                            if not self.agent.is_alive:
                                terminated = True
                                info['combat_outcome'] = "Agent died to Opportunity Attack"
                                # Skip agent's move if killed by OA
                                return self._get_obs(), reward, terminated, truncated, self._get_info()

                    if not self.agent.is_alive: # Check again if OA killed agent
                         action_taken_successfully = False # Agent couldn't complete move
                    elif [target_x, target_y] == self.enemy.position and self.enemy.is_alive:
                        action_taken_successfully = False
                        info['agent_move_outcome'] = "Blocked by enemy."
                    elif self._is_valid_position(target_x, target_y):
                        if self.agent.move(dx, dy, self.map_width, self.map_height): # move() deducts speed
                             action_taken_successfully = True
                             info['agent_move_outcome'] = f"Moved ({dx},{dy})."
                        else:
                            action_taken_successfully = False
                            info['agent_move_outcome'] = "Move failed unexpectedly (e.g. internal speed check failed)."
                    else:
                        # Invalid target position (out of bounds)
                        action_taken_successfully = False
                        info['agent_move_outcome'] = "Move out of bounds."
                else:
                    action_taken_successfully = False # Not enough speed for the move action itself
                    info['agent_move_outcome'] = "Not enough speed."

            elif decoded_action["type"] == "attack":
                if not self.agent.used_action_this_turn:
                    action_taken_successfully = True
                    attack_idx = decoded_action["index"]
                    selected_attack_stats = self.agent.attacks[attack_idx]
                    attack_name = selected_attack_stats.get("name", f"Attack {attack_idx}")
                    attack_range = selected_attack_stats.get("range", 1)
                    attack_type = selected_attack_stats.get("attack_type", "melee") # Assume melee if not specified
                    
                    adv_disadv_status = None
                    distance_to_enemy = self._calculate_manhattan_distance(self.agent.position, self.enemy.position)

                    # Ranged Attack Disadvantage
                    if attack_type == "ranged" and distance_to_enemy <= 1: # Assuming 1 cell is 5ft
                        adv_disadv_status = 'disadvantage'
                        info['agent_attack_modifier'] = "Ranged attack at disadvantage (enemy adjacent)."

                    # Dodge Effect on Target
                    if self.enemy.is_dodging:
                        if adv_disadv_status == 'advantage': # Advantage and disadvantage cancel
                            adv_disadv_status = None
                            info['agent_attack_modifier'] = info.get('agent_attack_modifier', "") + " Enemy dodging cancels advantage."
                        else: # No advantage, or already disadvantage
                            adv_disadv_status = 'disadvantage'
                            info['agent_attack_modifier'] = info.get('agent_attack_modifier', "") + " Enemy dodging, attack at disadvantage."

                    info['agent_action_details'] = f"Attempt Attack: {attack_name} (Range: {attack_range}, Type: {attack_type}, Adv/Disadv: {adv_disadv_status}), Target Dist: {distance_to_enemy}"

                    if distance_to_enemy <= attack_range:
                        old_enemy_hp = self.enemy.current_hp
                        # Pass adv_disadv_status to make_attack (implementation of dice rolling with adv/disadv is in make_attack)
                        hit, _ = self.agent.make_attack(self.enemy, attack_idx, self.dice_roller, advantage_disadvantage=adv_disadv_status)
                        actual_damage_dealt = old_enemy_hp - self.enemy.current_hp

                        if hit:
                            reward += actual_damage_dealt
                            info['agent_attack_outcome'] = f"Hit, dealt {actual_damage_dealt} damage."
                        else:
                            info['agent_attack_outcome'] = "Miss."

                        if not self.enemy.is_alive:
                            reward += 100 # Win reward
                            terminated = True
                            info['combat_outcome'] = "Agent won"
                    else:
                        info['agent_attack_outcome'] = "Out of range."
                    self.agent.used_action_this_turn = True
                else:
                    action_taken_successfully = False
                    info['agent_action_details'] = "Agent tried Attack but action already used."
                    info['agent_attack_outcome'] = "Action already used."
            
            elif decoded_action["type"] == "bonus_action":
                if not self.agent.used_bonus_action_this_turn:
                    ba_name = decoded_action["name"]
                    info['agent_action_details'] = f"Attempt Bonus Action: {ba_name}"
                    # Implement specific bonus action effects here
                    if ba_name == "bonus_move_1_cell": # Example
                        self.agent.speed_remaining += 1
                        reward += 0.02
                        action_taken_successfully = True
                        info['agent_bonus_outcome'] = "Used bonus_move_1_cell, gained 1 speed."
                    # Add other specific bonus actions here if needed
                    else:
                        # Generic placeholder for other bonus actions
                        action_taken_successfully = True
                        info['agent_bonus_outcome'] = f"Used bonus action: {ba_name} (effect placeholder)."

                    self.agent.used_bonus_action_this_turn = True
                else:
                    action_taken_successfully = False
                    info['agent_action_details'] = "Agent tried Bonus Action but it was already used."
                    info['agent_bonus_outcome'] = "Bonus action already used or not available."

            elif decoded_action["type"] == "dash":
                if not self.agent.used_action_this_turn:
                    self.agent.took_dash_action_this_turn = True
                    # Dash effectively increases available movement for the turn.
                    # Speed_remaining is increased by speed_total.
                    self.agent.speed_remaining += self.agent.speed_total
                    self.agent.used_action_this_turn = True
                    action_taken_successfully = True
                    reward += 0.05
                    info['agent_action_details'] = "Agent used Dash action."
                else:
                    action_taken_successfully = False
                    info['agent_action_details'] = "Agent tried Dash but action already used."

            elif decoded_action["type"] == "disengage":
                if not self.agent.used_action_this_turn:
                    self.agent.is_disengaging = True
                    self.agent.used_action_this_turn = True
                    action_taken_successfully = True
                    reward += 0.05
                    info['agent_action_details'] = "Agent used Disengage action."
                else:
                    action_taken_successfully = False
                    info['agent_action_details'] = "Agent tried Disengage but action already used."

            elif decoded_action["type"] == "dodge":
                if not self.agent.used_action_this_turn:
                    self.agent.is_dodging = True
                    self.agent.used_action_this_turn = True
                    action_taken_successfully = True
                    reward += 0.05
                    info['agent_action_details'] = "Agent used Dodge action."
                else:
                    action_taken_successfully = False
                    info['agent_action_details'] = "Agent tried Dodge but action already used."

            elif decoded_action["type"] == "end_turn":
                action_taken_successfully = True # Signals agent is done
                info['agent_action_details'] = "Agent chose to end turn."
                # The turn will now proceed to the enemy. All agent's remaining actions are forfeited.
                # To make this more explicit, we can set flags, though enemy turn follows regardless.
                self.agent.used_action_this_turn = True
                self.agent.used_bonus_action_this_turn = True
                self.agent.speed_remaining = 0


            elif decoded_action["type"] == "pass_turn":
                action_taken_successfully = True # Passing is a valid choice
                info['agent_action_details'] = "Agent passed turn (took no specific action)."
                # This implies forfeiting action, bonus action, and movement for this step.
                # If "pass_turn" means do absolutely nothing and end turn immediately:
                self.agent.used_action_this_turn = True
                self.agent.used_bonus_action_this_turn = True
                self.agent.speed_remaining = 0


            # If no action was successfully taken (e.g. tried to use an action already spent)
            if not action_taken_successfully:
                # Small penalty for trying an invalid sequence or running out of options.
                reward -= 0.05
                info['agent_action_outcome'] = info.get('agent_action_outcome', "Action failed or not allowed.")


        # If agent's action caused termination (e.g. enemy defeated by agent's attack)
        if terminated:
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        # Enemy's Turn
        if self.enemy.can_act(): # Check if enemy can act before its turn starts
            # Call start_new_turn() for the enemy at the beginning of its turn.
            self.enemy.start_new_turn()
            dist_to_agent = self._calculate_manhattan_distance(self.enemy.position, self.agent.position)

            if self.enemy_is_model_controlled and self.enemy_model and self.enemy_obs_flattener:
                info['enemy_ai_type'] = 'model_controlled'
                enemy_obs_dict = self._get_enemy_observation()
                flat_enemy_obs = self.enemy_obs_flattener.observation(enemy_obs_dict)
                enemy_action_id_array, _ = self.enemy_model.predict(flat_enemy_obs, deterministic=True)
                enemy_action_id = int(enemy_action_id_array.item()) # Convert numpy array to int
                
                decoded_enemy_action = self._decode_action(enemy_action_id, is_enemy=True)
                info['enemy_action_choice'] = f"Model chose: {decoded_enemy_action.get('type')} ({decoded_enemy_action.get('name', 'N/A')}; id {enemy_action_id})"
                
                enemy_action_type = decoded_enemy_action.get("type")
                enemy_action_param_idx = decoded_enemy_action.get("index") 
                enemy_action_delta = decoded_enemy_action.get("delta")

                # enemy_action_taken_successfully = False # Not currently used for reward for enemy model
                if enemy_action_type == "move":
                    if enemy_action_delta:
                        dx, dy = enemy_action_delta
                        cost_for_this_move_action = abs(dx) + abs(dy)
                        if self.enemy.speed_remaining >= cost_for_this_move_action:
                            enemy_old_pos = list(self.enemy.position)
                            enemy_target_x = self.enemy.position[0] + dx
                            enemy_target_y = self.enemy.position[1] + dy

                            # Opportunity Attack Check (Enemy Moving)
                            if self.agent.can_act() and \
                               self._calculate_manhattan_distance(enemy_old_pos, self.agent.position) == 1 and \
                               self._calculate_manhattan_distance([enemy_target_x, enemy_target_y], self.agent.position) > 1 and \
                               not self.enemy.is_disengaging and \
                               not self.agent.used_reaction_this_round:

                                agent_melee_attack_idx = -1
                                for i, attack in enumerate(self.agent.attacks):
                                    if attack.get("attack_type", "melee") == "melee":
                                        agent_melee_attack_idx = i
                                        break

                                if agent_melee_attack_idx != -1:
                                    info['opportunity_attack_by_agent'] = f"Agent Opportunity Attack on Enemy from {self.agent.position} to {enemy_old_pos} as enemy moves to {[enemy_target_x, enemy_target_y]}"
                                    oa_adv_disadv_agent = None # No adv/disadv for standard OA by agent
                                    hit, damage_dealt_to_enemy = self.agent.make_attack(self.enemy, agent_melee_attack_idx, self.dice_roller, advantage_disadvantage=oa_adv_disadv_agent)
                                    if hit:
                                        reward += damage_dealt_to_enemy # Positive reward for agent dealing damage
                                        info['opportunity_attack_by_agent_outcome'] = f"Hit, dealt {damage_dealt_to_enemy} damage."
                                    else:
                                        info['opportunity_attack_by_agent_outcome'] = "Miss."
                                    self.agent.used_reaction_this_round = True
                                    if not self.enemy.is_alive:
                                        terminated = True
                                        reward += 100 # Bonus for agent winning via OA
                                        info['combat_outcome'] = "Enemy died to Agent Opportunity Attack"
                                        return self._get_obs(), reward, terminated, truncated, self._get_info()

                            if not self.enemy.is_alive: # Check if OA killed enemy
                                info['enemy_move_outcome'] = "Enemy died before completing move."
                            elif [enemy_target_x, enemy_target_y] == self.agent.position and self.agent.is_alive:
                                info['enemy_move_outcome'] = "Blocked by agent."
                                # No speed deduction for bumping in 5e.
                            elif self._is_valid_position(enemy_target_x, enemy_target_y):
                                if self.enemy.move(dx, dy, self.map_width, self.map_height):
                                    info['enemy_move_outcome'] = f"Moved ({dx},{dy})."
                                else:
                                    info['enemy_move_outcome'] = "Move failed (e.g. internal speed check)."
                            else: # Hit wall
                                info['enemy_move_outcome'] = "Hit wall."
                                # No speed deduction for bumping wall in 5e.
                        else:
                             info['enemy_move_outcome'] = "Not enough speed for move action."
                    else: 
                        info['enemy_move_outcome'] = "Invalid move action from model (no delta)."

                elif enemy_action_type == "attack":
                    attack_idx = enemy_action_param_idx # This is from model's action choice
                    if attack_idx is not None and 0 <= attack_idx < len(self.enemy.attacks):
                        selected_attack_stats = self.enemy.attacks[attack_idx]
                        attack_name = selected_attack_stats.get("name", f"Attack {attack_idx}")
                        attack_range = selected_attack_stats.get("range", 1)
                        enemy_attack_type = selected_attack_stats.get("attack_type", "melee")

                        enemy_adv_disadv_status = None
                        # Ranged attack disadvantage for enemy
                        if enemy_attack_type == "ranged" and dist_to_agent <= 1:
                            enemy_adv_disadv_status = 'disadvantage'
                            info['enemy_attack_modifier'] = "Ranged attack at disadvantage (agent adjacent)."

                        # Agent dodging effect
                        if self.agent.is_dodging:
                            if enemy_adv_disadv_status == 'advantage':
                                enemy_adv_disadv_status = None
                                info['enemy_attack_modifier'] = info.get('enemy_attack_modifier', "") + " Agent dodging cancels advantage."
                            else:
                                enemy_adv_disadv_status = 'disadvantage'
                                info['enemy_attack_modifier'] = info.get('enemy_attack_modifier', "") + " Agent dodging, attack at disadvantage."

                        info['enemy_action_details'] = f"Model Attack: {attack_name} (Range: {attack_range}, Type: {enemy_attack_type}, Adv/Disadv: {enemy_adv_disadv_status}), Target Dist: {dist_to_agent}"
                        
                        if dist_to_agent <= attack_range:
                            old_agent_hp = self.agent.current_hp
                            hit, _ = self.enemy.make_attack(self.agent, attack_idx, self.dice_roller, advantage_disadvantage=enemy_adv_disadv_status)
                            actual_damage_taken = old_agent_hp - self.agent.current_hp
                            if hit:
                                reward -= actual_damage_taken
                                info['enemy_attack_outcome'] = f"Hit, dealt {actual_damage_taken} damage."
                            else:
                                info['enemy_attack_outcome'] = "Miss."
                            if not self.agent.is_alive:
                                reward -= 100; terminated = True; info['combat_outcome'] = "Enemy won (model)"
                        else:
                            info['enemy_attack_outcome'] = "Out of range."
                        self.enemy.used_action_this_turn = True # Mark action as used
                    else: 
                        info['enemy_attack_outcome'] = f"Invalid attack index {attack_idx} from model / Pass."
                
                elif enemy_action_type == "pass":
                    info['enemy_action_details'] = "Model chose Pass."
                
                elif enemy_action_type == "bonus_action":
                    ba_name_enemy = decoded_enemy_action.get("name") # Get name from decoded action
                    is_valid_bonus = False
                    if hasattr(self.enemy, 'bonus_actions'):
                         # Check if the string ba_name_enemy is in the list self.enemy.bonus_actions
                        if ba_name_enemy in self.enemy.bonus_actions:
                             is_valid_bonus = True
                    
                    if is_valid_bonus and ba_name_enemy == "bonus_move_1_cell":
                         self.enemy.speed_remaining +=1
                         info['enemy_bonus_outcome'] = "Used bonus_move_1_cell."
                    else:
                         info['enemy_bonus_outcome'] = f"Bonus action '{ba_name_enemy}' not implemented or not available for enemy."
                    info['enemy_action_details'] = f"Model chose Bonus Action: {ba_name_enemy}"


                # Fallback for unhandled or invalid actions from model if no specific outcome was logged
                if 'enemy_move_outcome' not in info and 'enemy_attack_outcome' not in info and \
                   'enemy_bonus_outcome' not in info and enemy_action_type != "pass":
                    info['enemy_action_details'] = info.get('enemy_action_details', "") + " Model chose unhandled or invalid action type, resulted in Pass."

            else: # Rule-based AI for enemy (if not model-controlled)
                info['enemy_ai_type'] = 'rule_based'
                # Basic Rule-Based AI for Enemy (can be expanded for new actions)
                # For now, it will try to attack if action available, then move.
                # It doesn't yet use Dash, Dodge, Disengage intelligently.

                if not self.enemy.used_action_this_turn and self.enemy.attacks:
                    enemy_attack_stats = self.enemy.attacks[0]
                    enemy_attack_range = enemy_attack_stats.get("range", 1)
                    enemy_attack_type = enemy_attack_stats.get("attack_type", "melee") # Assume melee

                    rb_adv_disadv_status = None
                    if enemy_attack_type == "ranged" and dist_to_agent <= 1:
                        rb_adv_disadv_status = 'disadvantage'
                        info['enemy_attack_modifier'] = "Ranged attack at disadvantage (agent adjacent)."

                    if self.agent.is_dodging:
                        if rb_adv_disadv_status == 'advantage':
                            rb_adv_disadv_status = None
                            info['enemy_attack_modifier'] = info.get('enemy_attack_modifier', "") + " Agent dodging cancels advantage."
                        else:
                            rb_adv_disadv_status = 'disadvantage'
                            info['enemy_attack_modifier'] = info.get('enemy_attack_modifier', "") + " Agent dodging, attack at disadvantage."

                    if dist_to_agent <= enemy_attack_range:
                        old_agent_hp = self.agent.current_hp
                        hit, _ = self.enemy.make_attack(self.agent, 0, self.dice_roller, advantage_disadvantage=rb_adv_disadv_status)
                        self.enemy.used_action_this_turn = True
                        info['enemy_action_details'] = f"Rule-based Attack: {self.enemy.attacks[0].get('name')} (Adv/Disadv: {rb_adv_disadv_status})"
                        if hit:
                            actual_damage_taken = old_agent_hp - self.agent.current_hp
                            reward -= actual_damage_taken
                            info['enemy_attack_outcome'] = f"Hit, dealt {actual_damage_taken} damage."
                        else:
                            info['enemy_attack_outcome'] = "Miss."
                        if not self.agent.is_alive:
                            terminated = True
                            reward -= 100
                            info['combat_outcome'] = "Enemy won (rule-based)"
                    else:
                         info['enemy_attack_outcome'] = "Out of range for rule-based attack."
                
                # Rule-based Enemy Movement (with OA check)
                if self.enemy.speed_remaining > 0 and (not self.enemy.used_action_this_turn or not self.enemy.attacks):
                    enemy_old_pos_rb = list(self.enemy.position)
                    # Simple move towards agent logic for rule-based enemy
                    dx_rb, dy_rb = 0, 0
                    if self.agent.position[0] > self.enemy.position[0]: dx_rb = 1
                    elif self.agent.position[0] < self.enemy.position[0]: dx_rb = -1
                    if self.agent.position[1] > self.enemy.position[1]: dy_rb = 1
                    elif self.agent.position[1] < self.enemy.position[1]: dy_rb = -1

                    # Try X movement first
                    if dx_rb != 0 and self.enemy.speed_remaining > 0:
                        enemy_target_x_rb, enemy_target_y_rb = self.enemy.position[0] + dx_rb, self.enemy.position[1]
                        # OA Check for rule-based enemy movement (X-axis)
                        if self.agent.can_act() and \
                           self._calculate_manhattan_distance(enemy_old_pos_rb, self.agent.position) == 1 and \
                           self._calculate_manhattan_distance([enemy_target_x_rb, enemy_target_y_rb], self.agent.position) > 1 and \
                           not self.enemy.is_disengaging and not self.agent.used_reaction_this_round:
                            agent_melee_idx_rb_x = -1
                            for i, att in enumerate(self.agent.attacks):
                                if att.get("attack_type", "melee") == "melee": agent_melee_idx_rb_x = i; break
                            if agent_melee_idx_rb_x != -1:
                                info['opportunity_attack_by_agent_rb_x'] = f"Agent OA on Enemy (X-move) from {self.agent.position} to {enemy_old_pos_rb}"
                                hit_oa_rb_x, dmg_oa_rb_x = self.agent.make_attack(self.enemy, agent_melee_idx_rb_x, self.dice_roller, advantage_disadvantage=None)
                                if hit_oa_rb_x: reward += dmg_oa_rb_x; info['opportunity_attack_by_agent_rb_x_outcome'] = f"Hit, {dmg_oa_rb_x} dmg."
                                else: info['opportunity_attack_by_agent_rb_x_outcome'] = "Miss."
                                self.agent.used_reaction_this_round = True
                                if not self.enemy.is_alive: terminated = True; reward += 100; info['combat_outcome'] = "Enemy died to Agent OA (RB X-move)"; return self._get_obs(), reward, terminated, truncated, self._get_info()
                        
                        if self.enemy.is_alive and not ([enemy_target_x_rb, enemy_target_y_rb] == self.agent.position and self.agent.is_alive) and self._is_valid_position(enemy_target_x_rb, enemy_target_y_rb):
                            if self.enemy.move(dx_rb, 0, self.map_width, self.map_height):
                                info["enemy_move_outcome"] = f"Moved ({dx_rb},0) (rule-based)."
                                enemy_old_pos_rb = list(self.enemy.position) # Update old_pos for Y move check
                            else: info["enemy_move_outcome"] = "X-Move failed (rule-based)."
                        elif ([enemy_target_x_rb, enemy_target_y_rb] == self.agent.position and self.agent.is_alive) : info["enemy_move_outcome"] = "Blocked by agent (X-move)."
                        else: info["enemy_move_outcome"] = "Invalid X-move or hit wall."


                    # Try Y movement if still has speed
                    if dy_rb != 0 and self.enemy.speed_remaining > 0 and self.enemy.is_alive:
                        enemy_target_x_rb, enemy_target_y_rb = self.enemy.position[0], self.enemy.position[1] + dy_rb
                        # OA Check for rule-based enemy movement (Y-axis)
                        if self.agent.can_act() and \
                           self._calculate_manhattan_distance(enemy_old_pos_rb, self.agent.position) == 1 and \
                           self._calculate_manhattan_distance([enemy_target_x_rb, enemy_target_y_rb], self.agent.position) > 1 and \
                           not self.enemy.is_disengaging and not self.agent.used_reaction_this_round:
                            agent_melee_idx_rb_y = -1
                            for i, att in enumerate(self.agent.attacks):
                                if att.get("attack_type", "melee") == "melee": agent_melee_idx_rb_y = i; break
                            if agent_melee_idx_rb_y != -1:
                                info['opportunity_attack_by_agent_rb_y'] = f"Agent OA on Enemy (Y-move) from {self.agent.position} to {enemy_old_pos_rb}"
                                hit_oa_rb_y, dmg_oa_rb_y = self.agent.make_attack(self.enemy, agent_melee_idx_rb_y, self.dice_roller, advantage_disadvantage=None)
                                if hit_oa_rb_y: reward += dmg_oa_rb_y; info['opportunity_attack_by_agent_rb_y_outcome'] = f"Hit, {dmg_oa_rb_y} dmg."
                                else: info['opportunity_attack_by_agent_rb_y_outcome'] = "Miss."
                                self.agent.used_reaction_this_round = True
                                if not self.enemy.is_alive: terminated = True; reward += 100; info['combat_outcome'] = "Enemy died to Agent OA (RB Y-move)"; return self._get_obs(), reward, terminated, truncated, self._get_info()

                        if self.enemy.is_alive and not ([enemy_target_x_rb, enemy_target_y_rb] == self.agent.position and self.agent.is_alive) and self._is_valid_position(enemy_target_x_rb, enemy_target_y_rb):
                            current_move_outcome = info.get("enemy_move_outcome", "")
                            if self.enemy.move(0, dy_rb, self.map_width, self.map_height):
                                info["enemy_move_outcome"] = current_move_outcome + f" Moved (0,{dy_rb}) (rule-based)."
                            else: info["enemy_move_outcome"] = current_move_outcome + " Y-Move failed (rule-based)."
                        elif ([enemy_target_x_rb, enemy_target_y_rb] == self.agent.position and self.agent.is_alive) : info["enemy_move_outcome"] = info.get("enemy_move_outcome","") + " Blocked by agent (Y-move)."
                        else: info["enemy_move_outcome"] = info.get("enemy_move_outcome","") + " Invalid Y-move or hit wall."

                    if not info.get("enemy_move_outcome"): # If no move outcome was logged
                         info["enemy_move_outcome"] = "No valid moves or decided not to move (rule-based)."
                    else:
                        info["enemy_move_outcome"] = "No speed remaining (rule-based)."

                # Enemy ends its turn by default after these checks
                self.enemy.used_action_this_turn = True
                self.enemy.used_bonus_action_this_turn = True # Assuming no bonus actions for rule-based for now


        # Check for truncation due to max steps
        if not terminated and self.current_episode_steps >= self.max_episode_steps:
            truncated = True
            # Reward for truncation is neutral; step penalty and survival (if applicable) already applied.

        return self._get_obs(), reward, terminated, truncated, self._get_info()

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
