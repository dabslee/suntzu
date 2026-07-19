from typing import List, Dict, Callable, Tuple, Optional, Any, Union
from collections import deque
import random
import re
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

# sb3-contrib (MaskablePPO) is optional; DQN checkpoints keep working without it.
try:
    from sb3_contrib import MaskablePPO
    MASKABLE_AVAILABLE = True
except ImportError:
    MASKABLE_AVAILABLE = False
    class MaskablePPO: pass


def load_model_any(path: str, device: str = 'auto'):
    """Load a saved policy zip regardless of algorithm (MaskablePPO or DQN)."""
    errors = []
    if MASKABLE_AVAILABLE:
        try:
            return MaskablePPO.load(path, device=device)
        except Exception as e:
            errors.append(f"MaskablePPO: {e}")
    if SB3_AVAILABLE:
        try:
            return DQN.load(path, device=device)
        except Exception as e:
            errors.append(f"DQN: {e}")
    raise RuntimeError(f"Could not load model from {path}: {'; '.join(errors) or 'SB3 not installed'}")


# Assuming dice.py is in the same directory or PYTHONPATH
try:
    from dice import roll
except ImportError:
    # Fallback for environments where dice.py might not be directly available
    # This allows the file to be parsed, but it will fail at runtime if roll is used.
    print("Warning: dice.py not found. DnDCombatEnv will not work without it.")
    def roll(dice_string: str) -> int:
        raise NotImplementedError("Dice roller not available.")

DEFAULT_ABILITIES = {"str": 10, "dex": 10, "con": 10, "int": 10, "wis": 10, "cha": 10}


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
                 initial_position: List[int],
                 abilities: Optional[Dict[str, int]] = None,
                 skills: Optional[Dict[str, int]] = None,
                 passive_perception: Optional[int] = None,
                 traits: Optional[List[str]] = None,
                 token_image: Optional[str] = None,
                 display_name: Optional[str] = None):
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

        # Ability scores, skills, senses, traits (SRD-style)
        self.abilities: Dict[str, int] = {**DEFAULT_ABILITIES, **(abilities or {})}
        self.skills: Dict[str, int] = skills or {}  # skill name -> total bonus (incl. proficiency)
        self.passive_perception: int = (passive_perception if passive_perception is not None
                                        else 10 + self.ability_modifier("wis"))
        self.traits: List[str] = traits or []

        # Presentation only (pygame rendering); no effect on game mechanics.
        self.token_image: Optional[str] = token_image
        self.display_name: str = display_name or name

        # Combat state attributes
        self.conditions: set = set()  # e.g. {"prone", "grappled", "hidden"}
        self.used_action_this_turn: bool = False
        self.used_bonus_action_this_turn: bool = False
        self.used_reaction_this_round: bool = False # Note: Reactions are typically once per ROUND, not turn.
        self.is_dodging: bool = False
        self.is_disengaging: bool = False
        self.took_dash_action_this_turn: bool = False
        # Multiattack: remaining attack rolls granted by taking the Attack
        # action this turn, and the melee/ranged lock ("two melee or two
        # ranged attacks") set by the first attack of a multiattack.
        self.attacks_remaining: int = 0
        self.attack_type_lock: Optional[str] = None

    def ability_modifier(self, ability: str) -> int:
        return (self.abilities.get(ability, 10) - 10) // 2

    def skill_modifier(self, skill: str, default_ability: str) -> int:
        """Skill bonus if proficient (from the stat block), else the raw ability mod."""
        if skill in self.skills:
            return self.skills[skill]
        return self.ability_modifier(default_ability)

    def saving_throw(self, ability: str, dc: int, dice_roller: Callable[[str], int]) -> bool:
        """Roll a saving throw. Returns True on success."""
        return dice_roller("1d20") + self.ability_modifier(ability) >= dc

    def has_condition(self, condition: str) -> bool:
        return condition in self.conditions


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

    def move(self, dx: int, dy: int, map_width: int, map_height: int, cost: Optional[int] = None) -> bool:
        if not self.can_act(): return False
        # Crawling while prone costs double; callers pass the adjusted cost.
        cost_of_movement = cost if cost is not None else abs(dx) + abs(dy)
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
        self.conditions = set()
        self.used_action_this_turn = False
        self.used_bonus_action_this_turn = False
        self.used_reaction_this_round = False
        self.is_dodging = False
        self.is_disengaging = False
        self.took_dash_action_this_turn = False
        self.attacks_remaining = 0
        self.attack_type_lock = None

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
        self.used_reaction_this_round = False
        # Multiattack slots do not carry over between turns.
        self.attacks_remaining = 0
        self.attack_type_lock = None
        # Note: conditions (prone, grappled, ...) intentionally persist across
        # turns; prone ends only via the stand_up action.


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
                 enemy_model_path: Optional[str] = None,
                 verbose: bool = True):
        super().__init__()

        self.map_width = map_width
        self.map_height = map_height
        self.grid_size = grid_size
        # Gate success-path setup chatter (enemy model loading etc.); errors
        # and warnings always print. Vectorized training silences all but the
        # first env copy so the log isn't repeated n_envs times.
        self.verbose = verbose
        
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
        
        self.CELL_SIZE = 64
        self.STATS_AREA_HEIGHT = 220  # Two stat lines plus the action log
        self.TEXT_COLOR = (200, 200, 200) # Light grey for text
        self.AGENT_COLOR = (70, 190, 90)   # Green (agent role)
        self.ENEMY_COLOR = (220, 80, 80)   # Red (enemy role)
        self.GRID_COLOR = (50, 50, 50)   # Dark Grey
        self.BACKGROUND_COLOR = (0, 0, 0) # Black

        # Play-by-play log of what each side did, shown in the render panel
        # and inspectable by runners (e.g. run_env.py prints new entries).
        self.event_log: deque = deque(maxlen=200)
        self._token_cache: Dict[Tuple[Optional[str], int], Optional["pygame.Surface"]] = {}

        if self.render_mode in ['human', 'rgb_array']:
            if pygame is None:
                raise ImportError("Pygame is not installed, but is required for 'human' or 'rgb_array' render modes. Please install it (e.g., pip install pygame).")

            pygame.font.init() # Initialize font module explicitly
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18) # For grid coordinates or smaller text
            self.label_font = pygame.font.Font(None, 16)  # Token name labels
            self.log_font = pygame.font.Font(None, 18)    # Action log lines

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
                try:
                    # Important: Pass device='auto' or 'cpu' if running in CPU-only environment
                    self.enemy_model = load_model_any(self.enemy_model_path, device='auto')
                    self.enemy_is_model_controlled = True

                    # Enemy sees itself as 'agent' and current agent as 'enemy'.
                    # This dummy env only provides the observation flattening
                    # structure; it never steps.
                    dummy_env_for_flattener = DnDCombatEnv(
                        map_width=self.map_width,
                        map_height=self.map_height,
                        agent_stats=self.enemy_stats_template,  # Enemy is the 'agent' in its own model
                        enemy_stats=self.agent_stats_template,  # Main agent is the 'enemy' from enemy model's POV
                        grid_size=self.grid_size,
                        render_mode=None, # No rendering for this dummy env
                        export_frames_path=None, # No frame export
                        enemy_model_path=None, # CRITICAL: No recursive model loading
                        verbose=False
                    )
                    self.enemy_obs_flattener = FlattenObservation(dummy_env_for_flattener)
                    if self.verbose:
                        print(f"Loaded enemy model {self.enemy_model_path} "
                              f"({type(self.enemy_model).__name__}); flattener ready.")
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

        # Build action maps. Both are always built so either combatant can be
        # driven at micro-action granularity (used by the internal enemy turn,
        # by a model-controlled enemy, and by run_env's spectator loop).
        self._agent_action_map = self._build_action_map_for_creature(self.agent)
        self.action_space = spaces.Discrete(len(self._agent_action_map))
        self._enemy_action_map = self._build_action_map_for_creature(self.enemy)
        # Backwards-compatible alias (older code referenced this name).
        self._enemy_action_map_for_validation = self._enemy_action_map
        
        # Define Observation Space
        max_coord = max(self.map_width, self.map_height) -1
        max_dist = self.map_width + self.map_height # Max Manhattan distance
        
        flag_box = lambda: spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "agent_hp_norm": flag_box(),
            "enemy_hp_norm": flag_box(),
            "agent_pos": spaces.Box(low=0, high=max_coord, shape=(2,), dtype=np.int32),
            "enemy_pos": spaces.Box(low=0, high=max_coord, shape=(2,), dtype=np.int32),
            "distance_to_enemy": spaces.Box(low=0, high=max_dist, shape=(1,), dtype=np.float32),
            "agent_speed_remaining_norm": flag_box(),
            "agent_used_action": flag_box(),
            "agent_used_bonus_action": flag_box(),
            "agent_used_reaction": flag_box(),
            "agent_is_dodging": flag_box(),
            "agent_attacks_remaining_norm": flag_box(),
            "agent_is_prone": flag_box(),
            "agent_is_hidden": flag_box(),
            "agent_is_grappled": flag_box(),
            "enemy_is_dodging": flag_box(),
            "enemy_is_prone": flag_box(),
            "enemy_is_hidden": flag_box(),
            "enemy_is_grappled": flag_box(),
        })
        
        # self.render_mode = 'ansi' # Default, can be set by user # render_mode is now an __init__ param

    def _build_observation(self, me: Creature, foe: Creature) -> Dict[str, Any]:
        """Observation from `me`'s perspective ('agent_*' keys describe `me`).

        Used for both the agent's obs and the model-controlled enemy's obs so
        the two perspectives are guaranteed to stay symmetric.
        """
        def flag(value: bool) -> np.ndarray:
            return np.array([1.0 if value else 0.0], dtype=np.float32)

        me_hp_norm = me.current_hp / me.max_hp if me.max_hp > 0 else 0.0
        foe_hp_norm = foe.current_hp / foe.max_hp if foe.max_hp > 0 else 0.0

        # Clamp speed normalization to ensure it doesn't exceed 1.0 due to bonuses
        raw_speed_norm = me.speed_remaining / me.speed_total if me.speed_total > 0 else 0.0

        max_attacks = max((a.get("num_attacks", 1) for a in me.attacks), default=1)
        attacks_norm = min(1.0, me.attacks_remaining / max_attacks) if max_attacks > 0 else 0.0

        dist = self._calculate_manhattan_distance(me.position, foe.position)

        return {
            "agent_hp_norm": np.array([me_hp_norm], dtype=np.float32),
            "enemy_hp_norm": np.array([foe_hp_norm], dtype=np.float32),
            "agent_pos": np.array(me.position, dtype=np.int32),
            "enemy_pos": np.array(foe.position, dtype=np.int32),
            "distance_to_enemy": np.array([dist], dtype=np.float32),
            "agent_speed_remaining_norm": np.array([min(1.0, raw_speed_norm)], dtype=np.float32),
            "agent_used_action": flag(me.used_action_this_turn),
            "agent_used_bonus_action": flag(me.used_bonus_action_this_turn),
            "agent_used_reaction": flag(me.used_reaction_this_round),
            "agent_is_dodging": flag(me.is_dodging),
            "agent_attacks_remaining_norm": np.array([attacks_norm], dtype=np.float32),
            "agent_is_prone": flag(me.has_condition("prone")),
            "agent_is_hidden": flag(me.has_condition("hidden")),
            "agent_is_grappled": flag(me.has_condition("grappled")),
            "enemy_is_dodging": flag(foe.is_dodging),
            "enemy_is_prone": flag(foe.has_condition("prone")),
            "enemy_is_hidden": flag(foe.has_condition("hidden")),
            "enemy_is_grappled": flag(foe.has_condition("grappled")),
        }

    def _get_obs(self) -> Dict[str, Any]:
        return self._build_observation(self.agent, self.enemy)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "agent_hp": self.agent.current_hp,
            "enemy_hp": self.enemy.current_hp,
            "agent_max_hp": self.agent.max_hp,
            "enemy_max_hp": self.enemy.max_hp,
            "distance": self._calculate_manhattan_distance(self.agent.position, self.enemy.position)
        }

    def _get_enemy_observation(self) -> Dict[str, Any]:
        # From the enemy's perspective: 'agent_*' keys describe the enemy
        # itself, 'enemy_*' keys describe the current agent.
        return self._build_observation(self.enemy, self.agent)

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
        self.event_log.append(f"===== Episode {self.current_episode_num}: "
                              f"{self._role_label('agent')} vs {self._role_label('enemy')} =====")

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

        # 4. Standard D&D Actions available to every creature
        action_map.append({"type": "dash", "name": "dash"})
        action_map.append({"type": "disengage", "name": "disengage"})
        action_map.append({"type": "dodge", "name": "dodge"})
        action_map.append({"type": "stand_up", "name": "stand_up"})
        action_map.append({"type": "hide", "name": "hide"})
        action_map.append({"type": "search", "name": "search"})
        # Special melee attacks (replace one attack of the Attack action)
        action_map.append({"type": "shove_prone", "name": "shove_prone"})
        action_map.append({"type": "shove_push", "name": "shove_push"})
        action_map.append({"type": "grapple", "name": "grapple"})
        action_map.append({"type": "escape_grapple", "name": "escape_grapple"})

        # 4b. Trait-driven bonus actions (e.g. goblin Nimble Escape)
        if "nimble_escape" in creature.traits:
            action_map.append({"type": "bonus_disengage", "name": "bonus_disengage"})
            action_map.append({"type": "bonus_hide", "name": "bonus_hide"})

        # 5. End Turn action (hands the round over to the opponent)
        action_map.append({"type": "end_turn", "name": "end_turn"})

        return action_map

    def _decode_action(self, action_int: int, is_enemy: bool = False) -> Dict[str, Any]:
        # Decode against the acting creature's own map (agent and enemy may
        # have different attacks/bonus actions/traits).
        map_to_use = self._enemy_action_map if is_enemy else self._agent_action_map

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

    def _has_bonus_actions(self, creature: Creature) -> bool:
        return bool(creature.bonus_actions) or "nimble_escape" in creature.traits

    def _turn_economy_exhausted(self, creature: Creature) -> bool:
        """True when the creature has nothing meaningful left to do this turn."""
        bonus_done = creature.used_bonus_action_this_turn or not self._has_bonus_actions(creature)
        return (creature.used_action_this_turn and bonus_done and
                creature.speed_remaining <= 0 and creature.attacks_remaining <= 0)

    def _compute_action_mask(self, actor: Creature, opponent: Creature,
                             action_map: List[Dict[str, Any]]) -> np.ndarray:
        """Boolean mask of which actions in `action_map` would currently be
        legal for `actor`. Mirrors the validity rules in
        _execute_micro_action; end_turn is always legal so the mask is never
        all-False."""
        distance = self._calculate_manhattan_distance(actor.position, opponent.position)
        move_cost = 2 if actor.has_condition("prone") else 1
        has_attack_slot = (not actor.used_action_this_turn) or actor.attacks_remaining > 0

        mask = np.zeros(len(action_map), dtype=bool)
        for i, action in enumerate(action_map):
            action_type = action["type"]
            if action_type == "move":
                dx, dy = action["delta"]
                target_x, target_y = actor.position[0] + dx, actor.position[1] + dy
                valid = (not actor.has_condition("grappled")
                         and actor.speed_remaining >= move_cost
                         and self._is_valid_position(target_x, target_y)
                         and not ([target_x, target_y] == opponent.position and opponent.is_alive))
            elif action_type == "attack":
                idx = action.get("index", -1)
                if not (0 <= idx < len(actor.attacks)) or not has_attack_slot:
                    valid = False
                else:
                    stats = actor.attacks[idx]
                    kind = stats.get("attack_type", "melee")
                    lock_ok = (not actor.used_action_this_turn
                               or actor.attack_type_lock is None
                               or kind == actor.attack_type_lock)
                    max_range = stats.get("range_long", stats.get("range", 1))
                    valid = lock_ok and distance <= max_range
            elif action_type == "bonus_action":
                valid = (not actor.used_bonus_action_this_turn
                         and 0 <= action.get("index", -1) < len(actor.bonus_actions))
            elif action_type in ("dash", "disengage", "dodge"):
                valid = not actor.used_action_this_turn
            elif action_type == "stand_up":
                valid = (actor.has_condition("prone")
                         and actor.speed_remaining >= max(1, actor.speed_total // 2))
            elif action_type == "hide":
                valid = (not actor.used_action_this_turn
                         and not actor.has_condition("hidden") and distance > 1)
            elif action_type == "search":
                valid = not actor.used_action_this_turn and opponent.has_condition("hidden")
            elif action_type in ("shove_prone", "shove_push", "grapple"):
                valid = distance <= 1 and has_attack_slot
            elif action_type == "escape_grapple":
                valid = actor.has_condition("grappled") and not actor.used_action_this_turn
            elif action_type == "bonus_disengage":
                valid = not actor.used_bonus_action_this_turn
            elif action_type == "bonus_hide":
                valid = (not actor.used_bonus_action_this_turn
                         and not actor.has_condition("hidden") and distance > 1)
            elif action_type == "end_turn":
                valid = True
            else:
                valid = False
            mask[i] = valid
        return mask

    def action_masks(self) -> np.ndarray:
        """sb3-contrib MaskablePPO hook: validity mask over the agent's actions."""
        return self._compute_action_mask(self.agent, self.enemy, self._agent_action_map)

    # --- Spectator API: drive either combatant one micro-action at a time ---
    #
    # step() bundles one agent micro-action with the opponent's *entire* turn,
    # which is right for RL training but means a viewer can only render one
    # frame for the whole opponent turn. These helpers let run_env drive both
    # sides symmetrically (one micro-action, one frame) without the training
    # asymmetry. 'prefix' is 'agent' (Agent 1 seat) or 'enemy' (Agent 2 seat).

    def _side(self, prefix: str) -> Tuple[Creature, Creature, List[Dict[str, Any]]]:
        if prefix == 'agent':
            return self.agent, self.enemy, self._agent_action_map
        return self.enemy, self.agent, self._enemy_action_map

    def creature_for(self, prefix: str) -> Creature:
        """The Creature occupying a seat ('agent' = Agent 1, 'enemy' = Agent 2)."""
        return self._side(prefix)[0]

    def micro_step_limit(self, prefix: str) -> int:
        """Safety cap on micro-actions in one creature's turn."""
        return self._max_micro_steps(self._side(prefix)[0])

    def observation_for(self, prefix: str) -> Dict[str, Any]:
        """Dict observation from a combatant's own perspective (same schema for
        both, so one FlattenObservation wrapper works for either side)."""
        return self._get_obs() if prefix == 'agent' else self._get_enemy_observation()

    def action_mask_for(self, prefix: str) -> np.ndarray:
        actor, target, amap = self._side(prefix)
        return self._compute_action_mask(actor, target, amap)

    def begin_turn(self, prefix: str) -> None:
        """Refresh a combatant's action economy at the start of its turn."""
        actor, _, _ = self._side(prefix)
        actor.start_new_turn()

    def spectator_step(self, prefix: str, action_id: int) -> Tuple[bool, bool, Dict[str, Any]]:
        """Execute and log a single micro-action for `prefix`. Returns
        (terminated, turn_over, decoded_action). Reuses the exact same combat
        resolution as step()/the internal enemy turn, so opportunity attacks,
        conditions, and win detection all behave identically."""
        actor, target, _ = self._side(prefix)
        self.current_episode_steps += 1
        decoded = self._decode_action(action_id, is_enemy=(prefix == 'enemy'))
        sub: Dict[str, Any] = {}
        _, terminated, success = self._execute_micro_action(actor, target, decoded, sub, prefix)
        self._log_micro_action(prefix, decoded, sub)
        turn_over = (terminated
                     or decoded["type"] == "end_turn"
                     or not success  # invalid/failed action ends the turn (avoids loops)
                     or self._turn_economy_exhausted(actor))
        return terminated, turn_over, decoded

    def _has_keen_senses(self, creature: Creature) -> bool:
        return any(t.startswith("keen") for t in creature.traits)

    def _attempt_hide(self, actor: Creature, target: Creature, info: Dict[str, Any], prefix: str) -> bool:
        """Roll Stealth against the target's passive Perception.

        Terrain does not exist yet, so hiding is abstracted: it may only be
        attempted while not adjacent to the enemy (checked by callers), and it
        shifts dice odds rather than concealing the hider's position.
        """
        stealth_mod = actor.skill_modifier("stealth", "dex")
        roll_result = self.dice_roller("1d20") + stealth_mod
        if roll_result >= target.passive_perception:
            actor.conditions.add("hidden")
            info[f'{prefix}_hide_outcome'] = (
                f"Hidden (Stealth {roll_result} vs passive Perception {target.passive_perception}).")
            return True
        info[f'{prefix}_hide_outcome'] = (
            f"Failed to hide (Stealth {roll_result} vs passive Perception {target.passive_perception}).")
        return False

    def _contested_check(self, initiator_mod: int, defender_mod: int) -> bool:
        """Contested ability check; the initiator rolls first. Ties go to the
        defender (the situation stays as it is, per the 5e contest rule)."""
        initiator_roll = self.dice_roller("1d20") + initiator_mod
        defender_roll = self.dice_roller("1d20") + defender_mod
        return initiator_roll > defender_roll

    def _best_escape_modifier(self, creature: Creature) -> int:
        """Shove/grapple defenders may use Athletics (STR) or Acrobatics (DEX)."""
        return max(creature.skill_modifier("athletics", "str"),
                   creature.skill_modifier("acrobatics", "dex"))

    def _push_away(self, pusher: Creature, target: Creature) -> bool:
        """Forced movement: push the target one cell directly away from the
        pusher (no movement cost to the target). Returns False if blocked."""
        diff_x = target.position[0] - pusher.position[0]
        diff_y = target.position[1] - pusher.position[1]
        if abs(diff_x) >= abs(diff_y) and diff_x != 0:
            delta = (1 if diff_x > 0 else -1, 0)
        elif diff_y != 0:
            delta = (0, 1 if diff_y > 0 else -1)
        else:
            return False
        new_x = target.position[0] + delta[0]
        new_y = target.position[1] + delta[1]
        if self._is_valid_position(new_x, new_y):
            target.position[0] = new_x
            target.position[1] = new_y
            return True
        return False

    def _update_proximity_conditions(self, info: Dict[str, Any]) -> None:
        """Distance-dependent state, re-checked after any position change:
        being adjacent reveals hidden creatures; a grapple ends when the
        grappler is no longer within reach (moving away releases it)."""
        distance = self._calculate_manhattan_distance(self.agent.position, self.enemy.position)
        pairs = ((self.agent, "agent"), (self.enemy, "enemy"))
        if distance <= 1:
            for creature, label in pairs:
                if creature.has_condition("hidden"):
                    creature.conditions.discard("hidden")
                    info[f'{label}_hidden_revealed'] = "Revealed by adjacency."
        else:
            for creature, label in pairs:
                if creature.has_condition("grappled"):
                    creature.conditions.discard("grappled")
                    info[f'{label}_grapple_released'] = "Grapple ended (grappler out of reach)."

    def _resolve_attack_modifiers(self, attacker: Creature, defender: Creature,
                                  attack_stats: Dict[str, Any], distance: int,
                                  attacker_label: str, defender_label: str) -> Tuple[Optional[str], List[str]]:
        """Collect every advantage/disadvantage source for an attack roll and
        apply the 5e stacking rule: if both kinds are present — no matter how
        many of each — the roll is straight.

        Returns (status, messages) where status is 'advantage',
        'disadvantage', or None.
        """
        advantage_sources: List[str] = []
        disadvantage_sources: List[str] = []
        attack_kind = attack_stats.get("attack_type", "melee")

        # Ranged attack with an adjacent hostile creature
        if attack_kind == "ranged" and distance <= 1:
            disadvantage_sources.append(f"Ranged attack at disadvantage ({defender_label} adjacent).")

        # Beyond normal range (up to long range)
        range_normal = attack_stats.get("range", 1)
        range_long = attack_stats.get("range_long")
        if attack_kind == "ranged" and range_long and distance > range_normal:
            disadvantage_sources.append("Attack at long range, disadvantage.")

        # Defender is dodging
        if defender.is_dodging:
            disadvantage_sources.append(f"{defender_label.capitalize()} dodging, attack at disadvantage.")

        # Defender is prone: advantage from within 5 ft, disadvantage beyond
        if defender.has_condition("prone"):
            if distance <= 1:
                advantage_sources.append(f"{defender_label.capitalize()} prone, attack at advantage (adjacent).")
            else:
                disadvantage_sources.append(f"{defender_label.capitalize()} prone, attack at disadvantage (not adjacent).")

        # Attacker is prone
        if attacker.has_condition("prone"):
            disadvantage_sources.append(f"{attacker_label.capitalize()} prone, attack at disadvantage.")

        # Attacker is hidden (unseen attacker) — hidden lands in the stealth phase
        if attacker.has_condition("hidden"):
            advantage_sources.append(f"{attacker_label.capitalize()} attacks from hiding, advantage.")
        # Defender is hidden (unseen target)
        if defender.has_condition("hidden"):
            disadvantage_sources.append(f"{defender_label.capitalize()} is hidden, attack at disadvantage.")

        # pack_tactics: needs an ally adjacent to the defender — inert in 1v1
        # combat, wired here so multi-creature support activates it.

        messages = advantage_sources + disadvantage_sources
        if advantage_sources and disadvantage_sources:
            messages.append("Advantage and disadvantage cancel out.")
            return None, messages
        if advantage_sources:
            return 'advantage', messages
        if disadvantage_sources:
            return 'disadvantage', messages
        return None, []

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
            # Crawling while prone costs 1 extra cell of movement per cell.
            cost = (abs(dx) + abs(dy)) * (2 if actor.has_condition("prone") else 1)
            if actor.has_condition("grappled"):
                info[f'{prefix}_move_outcome'] = "Cannot move while grappled."
            elif actor.speed_remaining < cost:
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
                            oa_stats = target.attacks[melee_idx]
                            oa_status, oa_messages = self._resolve_attack_modifiers(
                                target, actor, oa_stats, 1, other_prefix, prefix)
                            if oa_messages:
                                info[f'opportunity_attack_by_{other_prefix}_modifier'] = " ".join(oa_messages)
                            hp_before = actor.current_hp
                            hit, _ = target.make_attack(actor, melee_idx, self.dice_roller,
                                                        advantage_disadvantage=oa_status)
                            target.used_reaction_this_round = True
                            if target.has_condition("hidden"):
                                target.conditions.discard("hidden")
                                info[f'{other_prefix}_hidden_revealed'] = "Revealed by attacking."
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
                    if actor.move(dx, dy, self.map_width, self.map_height, cost=cost):
                        success = True
                        info[f'{prefix}_move_outcome'] = f"Moved ({dx},{dy})."
                        self._update_proximity_conditions(info)
                    else:
                        info[f'{prefix}_move_outcome'] = "Move failed unexpectedly."

        elif action_type == "attack":
            attack_idx = decoded_action["index"]
            attack_stats = actor.attacks[attack_idx]
            attack_kind = attack_stats.get("attack_type", "melee")

            if actor.used_action_this_turn and actor.attacks_remaining <= 0:
                info[f'{prefix}_action_details'] = f"{prefix.capitalize()} tried Attack but action already used."
                info[f'{prefix}_attack_outcome'] = "Action already used."
            elif (actor.used_action_this_turn and actor.attack_type_lock is not None
                  and attack_kind != actor.attack_type_lock):
                info[f'{prefix}_action_details'] = (
                    f"{prefix.capitalize()} Multiattack is locked to {actor.attack_type_lock} attacks this turn.")
                info[f'{prefix}_attack_outcome'] = "Wrong attack type for Multiattack."
            else:
                if not actor.used_action_this_turn:
                    # Taking the Attack action grants num_attacks attack rolls,
                    # spent one micro-action at a time (Multiattack). The
                    # melee/ranged type of the first attack locks the rest
                    # ("two melee attacks or two ranged attacks").
                    actor.used_action_this_turn = True
                    actor.attacks_remaining = attack_stats.get("num_attacks", 1)
                    actor.attack_type_lock = attack_kind if actor.attacks_remaining > 1 else None
                actor.attacks_remaining -= 1

                attack_name = attack_stats.get("name", f"Attack {attack_idx}")
                attack_range = attack_stats.get("range", 1)
                max_range = attack_stats.get("range_long", attack_range)
                distance = self._calculate_manhattan_distance(actor.position, target.position)

                adv_disadv, modifier_messages = self._resolve_attack_modifiers(
                    actor, target, attack_stats, distance, prefix, other_prefix)
                if modifier_messages:
                    info[f'{prefix}_attack_modifier'] = " ".join(modifier_messages)

                info[f'{prefix}_action_details'] = (
                    f"Attack: {attack_name} (Range: {attack_range}/{max_range}, Type: {attack_kind}, "
                    f"Adv/Disadv: {adv_disadv}), Target Dist: {distance}, Attacks left: {actor.attacks_remaining}")

                if distance <= max_range:
                    hp_before = target.current_hp
                    hit, _ = actor.make_attack(target, attack_idx, self.dice_roller,
                                               advantage_disadvantage=adv_disadv)
                    if actor.has_condition("hidden"):
                        # Attacking reveals you, hit or miss (advantage already applied)
                        actor.conditions.discard("hidden")
                        info[f'{prefix}_hidden_revealed'] = "Revealed by attacking."
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
                    elif hit:
                        self._apply_on_hit_effects(attack_stats, target, info, prefix, other_prefix)
                else:
                    info[f'{prefix}_attack_outcome'] = "Out of range."
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
                info[f'{prefix}_action_details'] = f"{prefix.capitalize()} used {action_type.capitalize()} action."

        elif action_type == "stand_up":
            if not actor.has_condition("prone"):
                info[f'{prefix}_action_details'] = f"{prefix.capitalize()} tried to stand up but is not prone."
            else:
                cost = max(1, actor.speed_total // 2)  # Standing costs half your speed
                if actor.speed_remaining >= cost:
                    actor.speed_remaining -= cost
                    actor.conditions.discard("prone")
                    success = True
                    info[f'{prefix}_action_details'] = f"{prefix.capitalize()} stood up (cost {cost} movement)."
                else:
                    info[f'{prefix}_action_details'] = (
                        f"{prefix.capitalize()} cannot stand up (needs {cost} movement, "
                        f"has {actor.speed_remaining}).")

        elif action_type == "hide":
            distance = self._calculate_manhattan_distance(actor.position, target.position)
            if actor.used_action_this_turn:
                info[f'{prefix}_action_details'] = f"{prefix.capitalize()} tried Hide but action already used."
            elif actor.has_condition("hidden"):
                info[f'{prefix}_action_details'] = f"{prefix.capitalize()} is already hidden."
            elif distance <= 1:
                info[f'{prefix}_action_details'] = (
                    f"{prefix.capitalize()} cannot hide while adjacent to the {other_prefix}.")
            else:
                actor.used_action_this_turn = True
                success = True  # the action is legal; the Stealth roll may still fail
                self._attempt_hide(actor, target, info, prefix)
                info[f'{prefix}_action_details'] = f"{prefix.capitalize()} used Hide action."

        elif action_type == "search":
            if actor.used_action_this_turn:
                info[f'{prefix}_action_details'] = f"{prefix.capitalize()} tried Search but action already used."
            elif not target.has_condition("hidden"):
                info[f'{prefix}_action_details'] = f"{prefix.capitalize()} searched but nothing is hidden."
            else:
                actor.used_action_this_turn = True
                success = True
                perception_mod = actor.skill_modifier("perception", "wis")
                # Keen senses (hearing/sight/smell) grant advantage on the check
                roll_str = "2d20kh1" if self._has_keen_senses(actor) else "1d20"
                roll_result = self.dice_roller(roll_str) + perception_mod
                dc = 10 + target.skill_modifier("stealth", "dex")
                if roll_result >= dc:
                    target.conditions.discard("hidden")
                    info[f'{prefix}_search_outcome'] = (
                        f"Found the {other_prefix} (Perception {roll_result} vs DC {dc}).")
                else:
                    info[f'{prefix}_search_outcome'] = (
                        f"Search failed (Perception {roll_result} vs DC {dc}).")
                info[f'{prefix}_action_details'] = f"{prefix.capitalize()} used Search action."

        elif action_type in ("shove_prone", "shove_push", "grapple"):
            verb = "Grapple" if action_type == "grapple" else "Shove"
            distance = self._calculate_manhattan_distance(actor.position, target.position)
            if distance > 1:
                info[f'{prefix}_action_details'] = (
                    f"{prefix.capitalize()} tried {verb} but the {other_prefix} is out of reach.")
            elif actor.used_action_this_turn and actor.attacks_remaining <= 0:
                info[f'{prefix}_action_details'] = f"{prefix.capitalize()} tried {verb} but action already used."
            else:
                # A special melee attack: replaces one attack of the Attack action.
                if not actor.used_action_this_turn:
                    actor.used_action_this_turn = True
                    actor.attacks_remaining = max((a.get("num_attacks", 1) for a in actor.attacks), default=1)
                actor.attacks_remaining -= 1
                success = True
                if actor.has_condition("hidden"):
                    actor.conditions.discard("hidden")
                    info[f'{prefix}_hidden_revealed'] = "Revealed by attacking."

                attacker_mod = actor.skill_modifier("athletics", "str")
                defender_mod = self._best_escape_modifier(target)
                info[f'{prefix}_action_details'] = (
                    f"{prefix.capitalize()} attempted {verb} "
                    f"(Athletics {attacker_mod:+d} vs best of Athletics/Acrobatics {defender_mod:+d}).")
                if self._contested_check(attacker_mod, defender_mod):
                    if action_type == "grapple":
                        target.conditions.add("grappled")
                        info[f'{prefix}_contest_outcome'] = f"Grapple succeeded: {other_prefix} is grappled."
                    elif action_type == "shove_prone":
                        target.conditions.add("prone")
                        info[f'{prefix}_contest_outcome'] = f"Shove succeeded: {other_prefix} is prone."
                    else:  # shove_push
                        if self._push_away(actor, target):
                            info[f'{prefix}_contest_outcome'] = (
                                f"Shove succeeded: {other_prefix} pushed to {target.position}.")
                            self._update_proximity_conditions(info)
                        else:
                            info[f'{prefix}_contest_outcome'] = (
                                f"Shove succeeded but the {other_prefix} has nowhere to go.")
                else:
                    info[f'{prefix}_contest_outcome'] = f"{verb} failed (contest lost)."

        elif action_type == "escape_grapple":
            if not actor.has_condition("grappled"):
                info[f'{prefix}_action_details'] = f"{prefix.capitalize()} tried to escape but is not grappled."
            elif actor.used_action_this_turn:
                info[f'{prefix}_action_details'] = f"{prefix.capitalize()} tried to escape but action already used."
            else:
                actor.used_action_this_turn = True
                success = True
                escaper_mod = self._best_escape_modifier(actor)
                grappler_mod = target.skill_modifier("athletics", "str")
                info[f'{prefix}_action_details'] = f"{prefix.capitalize()} tried to escape the grapple."
                if self._contested_check(escaper_mod, grappler_mod):
                    actor.conditions.discard("grappled")
                    info[f'{prefix}_contest_outcome'] = "Escaped the grapple."
                else:
                    info[f'{prefix}_contest_outcome'] = "Failed to escape the grapple."

        elif action_type in ("bonus_disengage", "bonus_hide"):
            distance = self._calculate_manhattan_distance(actor.position, target.position)
            if "nimble_escape" not in actor.traits:
                info[f'{prefix}_action_details'] = f"{prefix.capitalize()} lacks Nimble Escape."
            elif actor.used_bonus_action_this_turn:
                info[f'{prefix}_action_details'] = (
                    f"{prefix.capitalize()} tried Bonus Action but it was already used.")
            elif action_type == "bonus_hide" and actor.has_condition("hidden"):
                info[f'{prefix}_action_details'] = f"{prefix.capitalize()} is already hidden."
            elif action_type == "bonus_hide" and distance <= 1:
                info[f'{prefix}_action_details'] = (
                    f"{prefix.capitalize()} cannot hide while adjacent to the {other_prefix}.")
            else:
                actor.used_bonus_action_this_turn = True
                success = True
                if action_type == "bonus_disengage":
                    actor.is_disengaging = True
                    info[f'{prefix}_action_details'] = (
                        f"{prefix.capitalize()} used Nimble Escape: Disengage (bonus action).")
                else:
                    self._attempt_hide(actor, target, info, prefix)
                    info[f'{prefix}_action_details'] = (
                        f"{prefix.capitalize()} used Nimble Escape: Hide (bonus action).")

        elif action_type == "end_turn":
            success = True
            info[f'{prefix}_action_details'] = f"{prefix.capitalize()} ended turn."

        elif action_type == "pass":
            # Fallback from _decode_action for an index this creature doesn't have.
            info[f'{prefix}_action_details'] = (f"{prefix.capitalize()} chose an action invalid for this "
                                                f"creature; treated as pass.")

        return reward_delta, terminated, success

    def _apply_on_hit_effects(self, attack_stats: Dict[str, Any], target: Creature,
                              info: Dict[str, Any], prefix: str, other_prefix: str) -> None:
        """Resolve riders like the wolf bite's 'DC 11 STR save or prone'."""
        for effect in attack_stats.get("on_hit_effects", []):
            save_spec = effect.get("save")
            if not save_spec or not target.is_alive:
                continue
            ability = save_spec["ability"]
            dc = save_spec["dc"]
            if target.saving_throw(ability, dc, self.dice_roller):
                info[f'{prefix}_rider_outcome'] = (
                    f"{other_prefix.capitalize()} succeeded on DC {dc} {ability.upper()} save.")
            else:
                condition = effect.get("on_fail", {}).get("condition")
                if condition:
                    target.conditions.add(condition)
                info[f'{prefix}_rider_outcome'] = (
                    f"{other_prefix.capitalize()} failed DC {dc} {ability.upper()} save"
                    + (f" and is {condition}." if condition else "."))

    def _run_enemy_turn(self, info: Dict[str, Any]) -> Tuple[float, bool]:
        """Run the enemy's complete turn. Returns (reward_delta, terminated)."""
        self.enemy.start_new_turn()
        if self.enemy_is_model_controlled and self.enemy_model and self.enemy_obs_flattener:
            return self._run_enemy_model_turn(info)
        return self._run_enemy_rule_based_turn(info)

    def _run_enemy_micro(self, decoded: Dict[str, Any],
                         info: Dict[str, Any]) -> Tuple[float, bool, bool]:
        """Execute and log one enemy micro-action. Each call gets a fresh
        outcome dict so its log line reflects only this action (multiattack
        and multi-cell moves would otherwise overwrite each other in info),
        then merges into the step's info for the return value."""
        sub: Dict[str, Any] = {}
        r, terminated, success = self._execute_micro_action(self.enemy, self.agent, decoded, sub, 'enemy')
        info.update(sub)
        self._log_micro_action('enemy', decoded, sub)
        return r, terminated, success

    def _run_enemy_model_turn(self, info: Dict[str, Any]) -> Tuple[float, bool]:
        info['enemy_ai_type'] = 'model_controlled'
        reward_delta = 0.0
        terminated = False
        for _ in range(self._max_micro_steps(self.enemy)):
            enemy_obs_dict = self._get_enemy_observation()
            flat_enemy_obs = self.enemy_obs_flattener.observation(enemy_obs_dict)
            if MASKABLE_AVAILABLE and isinstance(self.enemy_model, MaskablePPO):
                enemy_mask = self._compute_action_mask(
                    self.enemy, self.agent, self._enemy_action_map_for_validation)
                enemy_action_id_array, _ = self.enemy_model.predict(
                    flat_enemy_obs, deterministic=True, action_masks=enemy_mask)
            else:
                enemy_action_id_array, _ = self.enemy_model.predict(flat_enemy_obs, deterministic=True)
            enemy_action_id = int(enemy_action_id_array.item())
            decoded = self._decode_action(enemy_action_id, is_enemy=True)

            r, terminated, success = self._run_enemy_micro(decoded, info)
            reward_delta += r
            if terminated or decoded["type"] == "end_turn" or self._turn_economy_exhausted(self.enemy):
                break
            if not success:
                # A failed action leaves the observation unchanged, so a
                # deterministic policy would repeat it forever; end the turn.
                break
        return reward_delta, terminated

    def _run_enemy_rule_based_turn(self, info: Dict[str, Any]) -> Tuple[float, bool]:
        info['enemy_ai_type'] = 'rule_based'
        reward_delta = 0.0
        terminated = False
        attack_stats = self.enemy.attacks[0] if self.enemy.attacks else None
        attack_range = attack_stats.get("range", 1) if attack_stats else 1

        # Stand up first if prone (costs half speed; crawling costs double).
        if self.enemy.has_condition("prone"):
            r, terminated, _ = self._run_enemy_micro({"type": "stand_up", "name": "stand_up"}, info)
            reward_delta += r

        # If the agent is hidden, spend the action searching for it rather
        # than attacking at disadvantage.
        if not terminated and self.agent.has_condition("hidden") and not self.enemy.used_action_this_turn:
            r, terminated, _ = self._run_enemy_micro({"type": "search", "name": "search"}, info)
            reward_delta += r

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
                r, terminated, success = self._run_enemy_micro(decoded, info)
                reward_delta += r
                if terminated:
                    return reward_delta, terminated
                if success:
                    moved = True
                    break
            if not moved:
                break  # Boxed in; stop trying to move.

        # Attack while in range, spending every Multiattack slot.
        while (not terminated and attack_stats and self.agent.is_alive and
               (not self.enemy.used_action_this_turn or self.enemy.attacks_remaining > 0) and
               self._calculate_manhattan_distance(self.enemy.position, self.agent.position) <= attack_range):
            decoded = {"type": "attack", "index": 0,
                       "name": f"attack_{attack_stats.get('name', '0')}"}
            r, terminated, success = self._run_enemy_micro(decoded, info)
            reward_delta += r
            if not success:
                break

        return reward_delta, terminated

    def _role_label(self, prefix: str) -> str:
        """Display name for a combatant by role. Both sides are framed
        symmetrically as 'Agent 1' (the agent seat) and 'Agent 2' (the enemy
        seat) — in self-play these are two policies, not a hero and a monster."""
        if prefix == 'agent':
            return f"Agent 1 ({self.agent.display_name})"
        return f"Agent 2 ({self.enemy.display_name})"

    _ROLE_WORD_RE = re.compile(r'\b(agent|enemy)\b', re.IGNORECASE)

    def _translate_roles(self, text: str) -> str:
        """Rewrite the internal role words 'agent'/'enemy' (any case) into the
        symmetric Agent 1 / Agent 2 display framing. Applied to outcome text
        only, never to the pre-built 'Agent 1 (Name)' label."""
        return self._ROLE_WORD_RE.sub(
            lambda m: "Agent 1" if m.group(0).lower() == "agent" else "Agent 2", text)

    def _log_micro_action(self, prefix: str, decoded_action: Dict[str, Any],
                          sub_info: Dict[str, Any]) -> None:
        """Append one play-by-play line for a single micro-action. Called once
        per micro-action for BOTH combatants, so the log reads identically for
        each side (one line per action, not a compressed turn summary)."""
        other = 'enemy' if prefix == 'agent' else 'agent'
        step_no = self.current_episode_steps
        name = decoded_action.get("name", "?")

        outcome_keys = (f'{prefix}_move_outcome', f'{prefix}_attack_outcome',
                        f'{prefix}_bonus_outcome', f'{prefix}_hide_outcome',
                        f'{prefix}_search_outcome', f'{prefix}_contest_outcome',
                        f'{prefix}_rider_outcome')
        parts = [sub_info[k] for k in outcome_keys if k in sub_info]
        if not parts and f'{prefix}_action_details' in sub_info:
            parts = [sub_info[f'{prefix}_action_details']]
        body = self._translate_roles(" ".join(parts)) if parts else "ok"
        self.event_log.append(f"[{step_no:>3}] {self._role_label(prefix)} {name}: {body}")

        # An opportunity attack resolves inside the mover's micro-action, by
        # the *other* combatant — log it as its own line under that side.
        oa_key = f'opportunity_attack_by_{other}_outcome'
        if oa_key in sub_info:
            self.event_log.append(
                f"[{step_no:>3}] {self._role_label(other)} opportunity attack: "
                f"{self._translate_roles(sub_info[oa_key])}")

        if 'combat_outcome' in sub_info:
            self.event_log.append(
                f"[{step_no:>3}] *** {self._translate_roles(sub_info['combat_outcome'])} ***")

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
            # Fresh outcome dict so this action logs on its own line (mirrors
            # how each enemy micro-action is logged), then merge into info.
            sub: Dict[str, Any] = {}
            reward_delta, terminated, success = self._execute_micro_action(
                self.agent, self.enemy, decoded_action, sub, 'agent')
            info.update(sub)
            self._log_micro_action('agent', decoded_action, sub)
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

    # Condition badges drawn on tokens: (condition, letter, color)
    CONDITION_BADGES = [("prone", "P", (230, 200, 60)),
                        ("hidden", "H", (110, 160, 255)),
                        ("grappled", "G", (240, 140, 40))]

    def _get_token_surface(self, creature: Creature, size: int) -> Optional["pygame.Surface"]:
        """Load and cache the creature's token image scaled to `size`."""
        key = (creature.token_image, size)
        if key in self._token_cache:
            return self._token_cache[key]
        surf = None
        if creature.token_image and os.path.exists(creature.token_image):
            try:
                img = pygame.image.load(creature.token_image)
                try:
                    img = img.convert_alpha()
                except pygame.error:
                    pass  # No display initialized (rgb_array mode) — raw surface is fine
                surf = pygame.transform.smoothscale(img, (size, size))
            except Exception as e:
                print(f"Warning: could not load token image {creature.token_image}: {e}")
        self._token_cache[key] = surf
        return surf

    def _draw_creature_token(self, surface: "pygame.Surface", creature: Creature,
                             border_color: Tuple[int, int, int]) -> None:
        cs = self.CELL_SIZE
        cell = pygame.Rect(creature.position[0] * cs, creature.position[1] * cs, cs, cs)
        label_h = 15
        token_size = cs - label_h - 6

        token = self._get_token_surface(creature, token_size)
        if token is not None:
            if creature.has_condition("hidden"):
                token = token.copy()
                token.set_alpha(110)  # ghosted while hidden
            surface.blit(token, (cell.x + (cs - token_size) // 2, cell.y + 3))
        else:
            # Fallback: colored square if the token image is missing
            pygame.draw.rect(surface, border_color,
                             pygame.Rect(cell.x + 5, cell.y + 5, cs - 10, cs - 10 - label_h))

        # Role border ring (green = agent, red = enemy)
        pygame.draw.rect(surface, border_color, cell, 2)

        # Small name label underneath the token
        label = self.label_font.render(creature.display_name, True, (235, 235, 235))
        label_bg = pygame.Rect(cell.x + 1, cell.bottom - label_h - 1, cs - 2, label_h)
        pygame.draw.rect(surface, (15, 15, 15), label_bg)
        surface.blit(label, (cell.centerx - label.get_width() // 2,
                             label_bg.y + (label_h - label.get_height()) // 2 + 1))

        # Condition badges along the top edge of the cell
        badges = [(ch, col) for cond, ch, col in self.CONDITION_BADGES
                  if creature.has_condition(cond)]
        if creature.is_dodging:
            badges.append(("D", (90, 220, 220)))
        bx = cell.x + 9
        for ch, col in badges:
            pygame.draw.circle(surface, col, (bx, cell.y + 9), 7)
            ch_surf = self.label_font.render(ch, True, (0, 0, 0))
            surface.blit(ch_surf, (bx - ch_surf.get_width() // 2,
                                   cell.y + 9 - ch_surf.get_height() // 2))
            bx += 16

    def _render_pygame_frame(self, surface: "pygame.Surface"):
        if surface is None:
            return

        surface.fill(self.BACKGROUND_COLOR)

        # Grid
        for x in range(self.map_width):
            for y in range(self.map_height):
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE,
                                   self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(surface, self.GRID_COLOR, rect, 1)

        # Creature tokens
        if self.enemy.is_alive:
            self._draw_creature_token(surface, self.enemy, self.ENEMY_COLOR)
        if self.agent.is_alive:
            self._draw_creature_token(surface, self.agent, self.AGENT_COLOR)

        # Stats + action-log panel
        panel_y = self.map_height * self.CELL_SIZE
        panel = pygame.Rect(0, panel_y, self.map_width * self.CELL_SIZE, self.STATS_AREA_HEIGHT)
        pygame.draw.rect(surface, (25, 25, 25), panel)

        def stat_line(creature: Creature, role_label: str) -> str:
            conditions = ",".join(sorted(creature.conditions)) if creature.conditions else "-"
            extras = []
            if creature.is_dodging:
                extras.append("dodging")
            if creature.attacks_remaining > 0:
                extras.append(f"{creature.attacks_remaining} atk left")
            extra_str = f"  [{', '.join(extras)}]" if extras else ""
            return (f"{role_label}: HP {creature.current_hp}/{creature.max_hp}  "
                    f"AC {creature.ac}  Spd {creature.speed_remaining}/{creature.speed_total}  "
                    f"Cond: {conditions}{extra_str}")

        surface.blit(self.font.render(stat_line(self.agent, self._role_label('agent')), True, self.AGENT_COLOR),
                     (10, panel_y + 8))
        surface.blit(self.font.render(stat_line(self.enemy, self._role_label('enemy')), True, self.ENEMY_COLOR),
                     (10, panel_y + 32))
        pygame.draw.line(surface, (60, 60, 60),
                         (10, panel_y + 56), (panel.right - 10, panel_y + 56))

        # Action log — most recent events at the bottom
        log_top = panel_y + 62
        line_height = 18
        max_lines = (self.STATS_AREA_HEIGHT - 62 - 6) // line_height
        for i, entry in enumerate(list(self.event_log)[-max_lines:]):
            text = self.log_font.render(entry[:92], True, self.TEXT_COLOR)
            surface.blit(text, (10, log_top + i * line_height))

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
