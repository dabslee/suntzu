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

    def take_damage(self, amount: int) -> None:
        if amount < 0: amount = 0
        self.current_hp -= amount
        if self.current_hp < 0: self.current_hp = 0
        if self.current_hp <= 0: self.is_alive = False

    def make_attack(self, target_creature: 'Creature', attack_index: int, dice_roller: Callable[[str], int]) -> Tuple[bool, int]:
        """ Returns (hit_status, damage_dealt) """
        if not self.can_act() or not target_creature.can_act():
            return False, 0 # (hit_status, damage_dealt)

        if not (0 <= attack_index < len(self.attacks)):
            raise IndexError(f"attack_index {attack_index} out of bounds for {self.name}'s attacks.")
            
        selected_attack = self.attacks[attack_index]
        attack_roll_result = dice_roller("1d20") + selected_attack["to_hit"]
        
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
            # "agent_can_attack": spaces.Discrete(2), # For simplicity, assume can attack if alive
            # "agent_can_bonus_action": spaces.Discrete(2) # For simplicity, assume can bonus if alive
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
            action_taken_successfully = False
            # Reset speed at start of agent's turn (or ensure it carries over if that's the design)
            # For now, assume speed is per turn and reset in self.agent.reset() for a new episode
            # If actions consume speed, it should persist within a turn.
            # Let's ensure speed is reset at the start of each agent's turn in the episode.
            # This is more like D&D 5e: you get your full speed each turn.
            self.agent.speed_remaining = self.agent.speed_total


            if decoded_action["type"] == "move":
                dx, dy = decoded_action["delta"]
                cost_for_this_move_action = 1 # Each "move direction" action attempts to move 1 cell costing 1 speed.
                
                if self.agent.speed_remaining >= cost_for_this_move_action:
                    target_x = self.agent.position[0] + dx
                    target_y = self.agent.position[1] + dy

                    if [target_x, target_y] == self.enemy.position:
                        # Cell is occupied by the enemy, move fails but speed is consumed for the attempt.
                        self.agent.speed_remaining -= cost_for_this_move_action
                        action_taken_successfully = True # Action was processed, though no position change.
                        # Optionally, add a small negative reward for bumping into enemy
                        # reward -= 0.01 
                    elif self._is_valid_position(target_x, target_y): # Check bounds before calling creature.move
                        # Creature.move will also check bounds and speed, but here we focus on occupation.
                        # The cost_for_this_move_action (1) is what this specific action costs.
                        # Creature.move itself will use its internal logic for speed deduction based on dx,dy.
                        # Since dx,dy is 1 cell, Creature.move will deduct 1 speed.
                        if self.agent.move(dx, dy, self.map_width, self.map_height):
                             action_taken_successfully = True
                        # If self.agent.move failed it means it hit a boundary or somehow still didn't have speed
                        # (though we checked speed_remaining >= cost_for_this_move_action).
                        # To ensure speed is consumed if Creature.move isn't called due to pre-check:
                        # This path assumes Creature.move is the sole speed deducter if target not occupied.
                    else:
                        # Invalid target position (out of bounds), but attempt still costs speed.
                        self.agent.speed_remaining -= cost_for_this_move_action
                        action_taken_successfully = True # Action processed, resulted in hitting a wall.
                else:
                    # Not enough speed for the move action itself
                    action_taken_successfully = False # Action could not be processed.

            elif decoded_action["type"] == "attack":
                action_taken_successfully = True # Attempting an attack is usually a successful action choice
                attack_idx = decoded_action["index"]
                selected_attack_stats = self.agent.attacks[attack_idx]
                attack_name = selected_attack_stats.get("name", f"Attack {attack_idx}")
                attack_range = selected_attack_stats.get("range", 1) # Default range to 1 if not specified
                
                distance_to_enemy = self._calculate_manhattan_distance(self.agent.position, self.enemy.position)
                info['agent_action_details'] = f"Attempt Attack: {attack_name} (Range: {attack_range}), Target Dist: {distance_to_enemy}"

                if distance_to_enemy <= attack_range:
                    old_enemy_hp = self.enemy.current_hp
                    hit, _ = self.agent.make_attack(self.enemy, attack_idx, self.dice_roller)
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
            
            elif decoded_action["type"] == "bonus_action":
                ba_name = decoded_action["name"]
                # Implement specific bonus action effects here
                if ba_name == "bonus_move_1_cell": # Example
                    # Allow moving 1 cell without using the standard move speed/action
                    # This is a conceptual example; how speed works with bonus moves needs care.
                    # For simplicity, let's say it gives +1 to current speed_remaining for this turn.
                    self.agent.speed_remaining += 1 
                    reward += 0.02 # Small reward for using a beneficial bonus action
                    action_taken_successfully = True
                elif ba_name == "quick_stab": # Another example
                    # Assume a predefined "quick_stab" is the last attack in the agent's attack list
                    # Or it needs its own definition {to_hit, damage_dice}
                    # For now, let's say it's a fixed small damage or a specific attack
                    # For simplicity, let's assume it's a conceptual action for now
                    # hit, damage = self.agent.make_attack(self.enemy, specific_bonus_attack_index, self.dice_roller)
                    # if hit: reward += damage * 0.05
                    # if not self.enemy.is_alive: terminated = True; reward += 50
                    action_taken_successfully = True # Using the bonus action
                    pass # Placeholder for more complex bonus actions

            elif decoded_action["type"] == "pass":
                action_taken_successfully = True # Passing is a valid choice
                pass

            # Agent survival reward for this turn (if action didn't end episode)
            # (The 'if not action_taken_successfully: pass' block was removed as it was neutral)
            # if self.agent.is_alive and not terminated:
            #     reward += 0.5

        # If agent's action caused termination (e.g. enemy defeated by agent's attack)
        if terminated:
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        # Enemy's Turn
        if self.enemy.can_act(): # Check if enemy can act before its turn starts
            self.enemy.speed_remaining = self.enemy.speed_total # Reset enemy speed for its turn
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
                        cost_for_this_move_action = 1 
                        if self.enemy.speed_remaining >= cost_for_this_move_action:
                            target_x = self.enemy.position[0] + dx
                            target_y = self.enemy.position[1] + dy
                            if [target_x, target_y] == self.agent.position: 
                                self.enemy.speed_remaining -= cost_for_this_move_action
                                info['enemy_move_outcome'] = "Blocked by agent."
                            elif self._is_valid_position(target_x, target_y): 
                                if self.enemy.move(dx, dy, self.map_width, self.map_height):
                                    info['enemy_move_outcome'] = f"Moved ({dx},{dy})."
                                else: 
                                     info['enemy_move_outcome'] = "Move failed (e.g. speed issue in Creature.move)."
                            else: # Hit wall
                                self.enemy.speed_remaining -= cost_for_this_move_action
                                info['enemy_move_outcome'] = "Hit wall."
                        else:
                             info['enemy_move_outcome'] = "Not enough speed for move action."
                    else: 
                        info['enemy_move_outcome'] = "Invalid move action from model (no delta)."

                elif enemy_action_type == "attack":
                    attack_idx = enemy_action_param_idx
                    if attack_idx is not None and 0 <= attack_idx < len(self.enemy.attacks):
                        selected_attack_stats = self.enemy.attacks[attack_idx]
                        attack_name = selected_attack_stats.get("name", f"Attack {attack_idx}")
                        attack_range = selected_attack_stats.get("range", 1)
                        
                        if dist_to_agent <= attack_range: 
                            info['enemy_action_details'] = f"Model Attack: {attack_name} (Range: {attack_range}), Target Dist: {dist_to_agent}"
                            old_agent_hp = self.agent.current_hp
                            hit, _ = self.enemy.make_attack(self.agent, attack_idx, self.dice_roller)
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
                enemy_attacked_this_turn = False
                # Rule-based AI: Attack if its first attack is in range
                if self.enemy.attacks:
                    enemy_attack_stats = self.enemy.attacks[0]
                    enemy_attack_range = enemy_attack_stats.get("range", 1)
                    if dist_to_agent <= enemy_attack_range : 
                        old_agent_hp = self.agent.current_hp
                        hit, _ = self.enemy.make_attack(self.agent, 0, self.dice_roller) 
                        enemy_attacked_this_turn = True
                        info['enemy_action_details'] = f"Rule-based Attack: {self.enemy.attacks[0].get('name')}"
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
                
                if not enemy_attacked_this_turn: 
                    current_action_detail = info.get('enemy_action_details', "") # Preserve potential "out of range" from attack attempt
                    if current_action_detail and not current_action_detail.endswith(" "): current_action_detail += " " 
                    info['enemy_action_details'] = current_action_detail + "Rule-based decided to move."
                    
                    enemy_move_cost = 1 
                    if self.enemy.speed_remaining >= enemy_move_cost:
                        dx, dy = 0, 0
                        if self.agent.position[0] > self.enemy.position[0]: dx = 1
                        elif self.agent.position[0] < self.enemy.position[0]: dx = -1
                        if self.agent.position[1] > self.enemy.position[1]: dy = 1
                        elif self.agent.position[1] < self.enemy.position[1]: dy = -1
                        
                        moved_enemy = False
                        # X-axis movement attempt
                        if dx != 0:
                            target_x = self.enemy.position[0] + dx
                            target_y = self.enemy.position[1]
                            is_target_agent_occupied = ([target_x, target_y] == self.agent.position)
                            is_target_map_valid = self._is_valid_position(target_x, target_y)
                            can_move_to_target = is_target_map_valid and not is_target_agent_occupied
                            if can_move_to_target:
                                if self.enemy.move(dx, 0, self.map_width, self.map_height):
                                    moved_enemy = True
                            if not can_move_to_target: 
                                self.enemy.speed_remaining -= enemy_move_cost 
                        
                        # Y-axis movement attempt
                        if not moved_enemy and dy != 0:
                            target_x = self.enemy.position[0]
                            target_y = self.enemy.position[1] + dy
                            is_target_agent_occupied = ([target_x, target_y] == self.agent.position)
                            is_target_map_valid = self._is_valid_position(target_x, target_y)
                            can_move_to_target = is_target_map_valid and not is_target_agent_occupied
                            if can_move_to_target:
                                if self.enemy.move(0, dy, self.map_width, self.map_height):
                                    moved_enemy = True 
                            if not can_move_to_target:
                                self.enemy.speed_remaining -= enemy_move_cost
                        
                        if moved_enemy: info["enemy_move_outcome"] = "Moved (rule-based)."
                        else: info["enemy_move_outcome"] = "Move attempt failed or blocked (rule-based)."

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
            {"name": "sword", "to_hit": 5, "damage_dice": "1d8+3", "num_attacks": 1},
            {"name": "dagger_offhand", "to_hit": 5, "damage_dice": "1d4+3", "num_attacks": 1}
        ],
        "bonus_actions": ["bonus_move_1_cell"] 
        # "initial_position" is not needed here as env sets it in reset
    }

    enemy_example_stats = {
        "ac": 13,
        "max_hp": 15,
        "speed_total": 6,
        "attacks": [
            {"name": "scimitar", "to_hit": 4, "damage_dice": "1d6+2", "num_attacks": 1}
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
