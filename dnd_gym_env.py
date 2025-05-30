from typing import List, Dict, Callable, Tuple, Optional, Any
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

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
            return False, 0

        if not (0 <= attack_index < len(self.attacks)):
            # This case should ideally be prevented by action space construction
            raise IndexError(f"attack_index {attack_index} out of bounds for {self.name}'s attacks.")
            
        selected_attack = self.attacks[attack_index]
        attack_roll_result = dice_roller("1d20") + selected_attack["to_hit"]
        
        damage_dealt = 0
        hit = False
        if attack_roll_result >= target_creature.ac:
            hit = True
            damage_dealt = dice_roller(selected_attack["damage_dice"])
            if damage_dealt < 0: damage_dealt = 0 
            target_creature.take_damage(damage_dealt)
        return hit, damage_dealt

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
    metadata = {'render_modes': ['ansi'], 'render_fps': 4}

    def __init__(self, map_width: int, map_height: int, agent_stats: dict, enemy_stats: dict, grid_size: int = 1):
        super().__init__()

        self.map_width = map_width
        self.map_height = map_height
        self.grid_size = grid_size # Feet per grid cell. Assume creature speeds are in cells for now.

        # Ensure 'initial_position' is not passed directly if it's part of agent_stats to avoid TypeError
        # The actual initial position will be set in reset()
        agent_stats_copy = agent_stats.copy()
        agent_stats_copy.pop('initial_position', None) 
        self.agent = Creature(name="agent", initial_position=[0,0], **agent_stats_copy)

        enemy_stats_copy = enemy_stats.copy()
        enemy_stats_copy.pop('initial_position', None)
        self.enemy = Creature(name="enemy", initial_position=[0,0], **enemy_stats_copy)
        
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
        
        self.render_mode = 'ansi' # Default, can be set by user

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

        return self._get_obs(), self._get_info()

    def _calculate_manhattan_distance(self, pos1: List[int], pos2: List[int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_valid_position(self, x: int, y: int) -> bool:
        return 0 <= x < self.map_width and 0 <= y < self.map_height

    @property # Make it accessible like a variable but defined as a method
    def action_map(self):
        return self._action_map

    def _decode_action(self, action_int: int) -> Dict[str, Any]:
        if not (0 <= action_int < len(self.action_map)):
            raise ValueError(f"Invalid action integer: {action_int}")
        return self.action_map[action_int]

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        terminated = False
        truncated = False # Not used for now, could be for time limits
        reward = -0.05 # Small penalty for existing / taking a turn

        decoded_action = self._decode_action(action)

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
                # For now, assume 1 unit of speed per 1 cell move action
                if self.agent.speed_remaining >= 1: # Cost of this move action
                    if self.agent.move(dx, dy, self.map_width, self.map_height):
                        action_taken_successfully = True
                        # Simple reward for moving: if it changes distance to enemy
                        # This is very basic, could be improved
                        # reward += 0.01 # Small reward for successful move
                # Note: self.agent.move already deducts speed. Here, we assume a "move action" uses 1 "action point"
                # and allows movement up to creature's speed. The current setup is 1 cell per move action.
                # This needs refinement if speed is to be a resource spent across multiple move actions.
                # For now: one "move N/S/E/W" action costs 1 speed and moves 1 cell.

            elif decoded_action["type"] == "attack":
                attack_idx = decoded_action["index"]
                # Check distance for melee/ranged attacks - future improvement
                # For now, assume any attack can be attempted.
                hit, damage = self.agent.make_attack(self.enemy, attack_idx, self.dice_roller)
                action_taken_successfully = True # Attempting an attack is an action
                if hit:
                    reward += damage * 0.1 # Reward scaled by damage
                if not self.enemy.is_alive:
                    terminated = True
                    reward += 50 # Large reward for defeating enemy
            
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

            if not action_taken_successfully:
                reward -= 0.1 # Penalty for trying an invalid/impossible action (e.g. move with 0 speed)

        if terminated:
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        # Enemy's Turn (Simple AI)
        if self.enemy.can_act():
            self.enemy.speed_remaining = self.enemy.speed_total # Reset enemy speed for its turn

            dist_to_agent = self._calculate_manhattan_distance(self.enemy.position, self.agent.position)
            
            if dist_to_agent == 1: # Adjacent (Manhattan distance for cardinal, not diagonal)
                if self.enemy.attacks: # Check if enemy has any attacks
                    # Enemy attacks agent (e.g., its first available attack)
                    hit, damage = self.enemy.make_attack(self.agent, 0, self.dice_roller)
                    if hit:
                        reward -= damage * 0.1 # Penalty scaled by damage taken
                    if not self.agent.is_alive:
                        terminated = True
                        reward -= 50 # Large penalty for agent defeat
            else:
                # Move enemy one step towards the agent
                dx, dy = 0, 0
                if self.agent.position[0] > self.enemy.position[0]: dx = 1
                elif self.agent.position[0] < self.enemy.position[0]: dx = -1
                
                if self.agent.position[1] > self.enemy.position[1]: dy = 1
                elif self.agent.position[1] < self.enemy.position[1]: dy = -1

                # Try to move in x, then y, or prefer non-diagonal if stuck
                moved = False
                if dx != 0 and self.enemy.move(dx, 0, self.map_width, self.map_height):
                    moved = True
                elif dy != 0 and self.enemy.move(0, dy, self.map_width, self.map_height):
                    moved = True
                
                # If stuck (e.g. dx=0 and dy move failed, or dx move failed and dy=0) try other axis if available
                if not moved:
                    if dx == 0 and dy != 0: # Was trying to move only in y, failed.
                        pass # No other primary axis to try
                    elif dy == 0 and dx != 0: # Was trying to move only in x, failed.
                        pass # No other primary axis to try
                    elif dx != 0 and dy != 0: # Was trying diagonal, one axis failed
                        if not self.enemy.move(dx,0, self.map_width, self.map_height): # Try x first
                           self.enemy.move(0,dy, self.map_width, self.map_height) # Then y


        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> Optional[str]:
        if self.render_mode is None: # For gym v26, render_mode can be None
            gym.logger.warn("Cannot render without specifying a render mode. Using 'ansi'.")
            self.render_mode = 'ansi'

        if self.render_mode == 'ansi':
            grid = [['.' for _ in range(self.map_width)] for _ in range(self.map_height)]
            
            # Place enemy
            if self.enemy.is_alive and self._is_valid_position(self.enemy.position[0], self.enemy.position[1]):
                grid[self.enemy.position[1]][self.enemy.position[0]] = 'E'
            
            # Place agent (overwrites enemy if same spot, though reset tries to avoid this)
            if self.agent.is_alive and self._is_valid_position(self.agent.position[0], self.agent.position[1]):
                grid[self.agent.position[1]][self.agent.position[0]] = 'A'

            ansi_str = "\n".join(" ".join(row) for row in grid)
            ansi_str += (f"\nAgent HP: {self.agent.current_hp}/{self.agent.max_hp}, Speed: {self.agent.speed_remaining}/{self.agent.speed_total} "
                         f"| Enemy HP: {self.enemy.current_hp}/{self.enemy.max_hp}")
            print(ansi_str)
            return ansi_str
        else:
            return None # Or raise error for unsupported modes

    def close(self):
        # Clean up any resources if needed
        pass

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
                       enemy_stats=enemy_example_stats)

    obs, info = env.reset()
    print("Initial Observation:", obs)
    print("Initial Info:", info)
    env.render()

    # Example: Agent tries to attack enemy (action index for first attack is 4: N,S,E,W, Attack0)
    # Find the sword attack index:
    sword_attack_action_idx = -1
    for i, action_def in enumerate(env.action_map):
        if action_def["type"] == "attack" and action_def["name"] == "attack_sword":
            sword_attack_action_idx = i
            break
    
    if sword_attack_action_idx != -1:
        print(f"\nTaking action: Attack with Sword (action index {sword_attack_action_idx})")
        obs, reward, terminated, truncated, info = env.step(sword_attack_action_idx)
        print("Observation after attack:", obs)
        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Info:", info)
        env.render()
    else:
        print("Could not find sword attack action for example.")

    # Example: Agent tries to move East (action index 2)
    move_east_action_idx = -1
    for i, action_def in enumerate(env.action_map):
        if action_def["type"] == "move" and action_def["name"] == "move_E":
            move_east_action_idx = i
            break
    if move_east_action_idx != -1:
        print(f"\nTaking action: Move East (action index {move_east_action_idx})")
        obs, reward, terminated, truncated, info = env.step(move_east_action_idx) # Move East
        print("Observation after move:", obs)
        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Info:", info)
        env.render()
    else:
        print("Could not find move East action for example.")
