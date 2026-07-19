"""The Creature data model: stats, action economy, conditions, and dice-facing
helpers (attack rolls, saving throws). No Gym, SB3, or pygame dependencies.
"""
from typing import List, Dict, Callable, Tuple, Optional

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

