from typing import List, Dict, Callable, Tuple, Optional, Any, Union
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Pygame import (optional)
try:
    import pygame
except ImportError:
    pygame = None # Flag to indicate Pygame is not available

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

    def __init__(self, map_width: int, map_height: int, agent_stats: dict, enemy_stats: dict, grid_size: int = 1, render_mode: Optional[str] = None):
        super().__init__()

        self.map_width = map_width
        self.map_height = map_height
        self.grid_size = grid_size # Feet per grid cell. Assume creature speeds are in cells for now.
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.render_surface = None # For rgb_array
        self.window_surface = None # Surface to draw on (screen or render_surface)
        self.max_episode_steps = 100
        
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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_episode_steps = 0

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
        self.current_episode_steps += 1
        terminated = False
        truncated = False 
        reward = -0.1  # Overall step penalty

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
                attack_idx = decoded_action["index"]
                # Check distance for melee/ranged attacks - future improvement
                # For now, assume any attack can be attempted.
                old_enemy_hp = self.enemy.current_hp
                hit, _ = self.agent.make_attack(self.enemy, attack_idx, self.dice_roller) # Raw damage roll not used directly for reward
                action_taken_successfully = True # Attempting an attack is an action
                if hit:
                    actual_damage_dealt = old_enemy_hp - self.enemy.current_hp # Calculate actual HP lost
                    reward += actual_damage_dealt
                if not self.enemy.is_alive:
                    terminated = True
                    reward += 100 # Win reward
            
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
                # No additional penalty here beyond the initial -0.1 step penalty.
                pass
            
            # Agent survival reward for this turn (if action didn't end episode)
            if self.agent.is_alive and not terminated:
                reward += 0.5

        # If agent's action caused termination (e.g. enemy defeated by agent's attack)
        if terminated:
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        # Enemy's Turn
        if self.enemy.can_act(): # Check if enemy can act before its turn starts
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
                # If adjacent but no attacks (or chose not to attack), then consider moving.
                # This else is now tied to 'if self.enemy.attacks:'
                # If there are no attacks, it will fall through to the movement logic below.
                # To make it explicit: if it didn't attack (either no attacks or other reason), it might move.
                # For simplicity, let's assume if it's adjacent and can't/doesn't attack, it will use the move logic.
                # This requires restructuring slightly: the move logic should be callable if not attacking OR not adjacent.

            # Unified movement logic for enemy if it didn't attack (or isn't adjacent)
            # Condition: if it's not adjacent OR (it is adjacent AND it did not attack)
            # The 'did not attack' part is tricky if attacks list is empty.
            # Let's simplify: if adjacent and has attacks, it attacks. Otherwise, it tries to move.
            
            enemy_attacked_this_turn = False
            if dist_to_agent == 1 and self.enemy.attacks:
                old_agent_hp = self.agent.current_hp
                hit, _ = self.enemy.make_attack(self.agent, 0, self.dice_roller) # Raw damage roll not used directly
                enemy_attacked_this_turn = True
                if hit:
                    actual_damage_taken = old_agent_hp - self.agent.current_hp # Calculate actual HP lost
                    reward -= actual_damage_taken
                if not self.agent.is_alive:
                    terminated = True
                    reward -= 100 # Loss penalty
            
            if not enemy_attacked_this_turn: # Move if not adjacent OR if adjacent but chose not/could not attack
                # Cost for enemy's 1-cell move attempt
                enemy_move_cost = 1 
                if self.enemy.speed_remaining >= enemy_move_cost:
                    dx, dy = 0, 0
                    if self.agent.position[0] > self.enemy.position[0]: dx = 1
                    elif self.agent.position[0] < self.enemy.position[0]: dx = -1
                    
                    if self.agent.position[1] > self.enemy.position[1]: dy = 1
                    elif self.agent.position[1] < self.enemy.position[1]: dy = -1

                    # Attempt to move towards agent, check occupation
                    # Try x-axis move first
                    enemy_target_x_try1 = self.enemy.position[0] + dx
                    enemy_target_y_try1 = self.enemy.position[1]
                    
                    moved_enemy = False
                    # X-axis movement attempt
                    if dx != 0:
                        target_x = self.enemy.position[0] + dx
                        target_y = self.enemy.position[1]
                        
                        is_target_agent_occupied = ([target_x, target_y] == self.agent.position)
                        is_target_map_valid = self._is_valid_position(target_x, target_y)

                        can_move_to_target = is_target_map_valid and not is_target_agent_occupied
                        if can_move_to_target:
                            if self.enemy.move(dx, 0, self.map_width, self.map_height): # move() deducts speed
                                moved_enemy = True
                        
                        if not can_move_to_target: # Collision with wall OR agent
                            self.enemy.speed_remaining -= enemy_move_cost 
                    
                    # Y-axis movement attempt (only if no x-move occurred or dx was 0)
                    if not moved_enemy and dy != 0:
                        target_x = self.enemy.position[0]
                        target_y = self.enemy.position[1] + dy

                        is_target_agent_occupied = ([target_x, target_y] == self.agent.position)
                        is_target_map_valid = self._is_valid_position(target_x, target_y)
                        
                        can_move_to_target = is_target_map_valid and not is_target_agent_occupied
                        if can_move_to_target:
                            if self.enemy.move(0, dy, self.map_width, self.map_height): # move() deducts speed
                                moved_enemy = True 
                        
                        if not can_move_to_target: # Collision with wall OR agent
                            self.enemy.speed_remaining -= enemy_move_cost
        
        # Truncate condition
        if self.current_episode_steps >= self.max_episode_steps and not terminated:
            truncated = True
            reward = 0

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
