"""The combat rules engine: pure D&D mechanics, no RL.

`CombatRules` owns a list of creatures (indexed, N-creature-ready — today's
combats have two) and resolves micro-actions: movement with opportunity
attacks, attacks with Multiattack slots and advantage/disadvantage stacking,
on-hit save riders, conditions (prone/hidden/grappled), stealth, and
contested checks (shove/grapple).

Deliberately absent from this module:
- Rewards. `execute()` reports what happened (`ActionResult.damage`/`deaths`)
  and the Gym adapter in sunztu.env decides what it is worth to the learner.
- Seats. There is no 'agent' or 'enemy' here — creatures are indices. Each
  creature has a display *label* used only to format human-readable outcome
  text (the env passes 'agent'/'enemy' so existing log/info wording is
  stable).
- Observations, pygame, SB3.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sunztu.creature import Creature
from sunztu.dice import roll


@dataclass
class ActionResult:
    """What one micro-action did.

    notes: label-prefixed human-readable fragments (e.g. 'agent_attack_outcome')
    damage: (attacker_idx, victim_idx, hp_lost) per damage instance
    deaths: creature indices that died during this action
    """
    success: bool = False
    notes: Dict[str, str] = field(default_factory=dict)
    damage: List[Tuple[int, int, int]] = field(default_factory=list)
    deaths: List[int] = field(default_factory=list)


class CombatRules:
    def __init__(self, creatures: List[Creature], labels: List[str],
                 map_width: int, map_height: int, dice_roller=roll):
        assert len(creatures) == len(labels)
        self.creatures = creatures
        self.labels = labels
        self.map_width = map_width
        self.map_height = map_height
        self.dice_roller = dice_roller
        self.action_maps = [self._build_action_map(c) for c in creatures]

    # --- Geometry ---------------------------------------------------------

    def distance_between(self, pos1: List[int], pos2: List[int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def distance(self, i: int, j: int) -> int:
        return self.distance_between(self.creatures[i].position, self.creatures[j].position)

    def is_valid_position(self, x: int, y: int) -> bool:
        return 0 <= x < self.map_width and 0 <= y < self.map_height

    # --- Roster -----------------------------------------------------------

    def opponent_of(self, idx: int) -> int:
        """The single hostile creature. Multi-creature combat will replace
        this with a target-selection seam; every current caller goes through
        here so that change stays local."""
        return 1 - idx

    def combat_over(self) -> bool:
        return any(not c.is_alive for c in self.creatures)

    # --- Action space -----------------------------------------------------

    def _build_action_map(self, creature: Creature) -> List[Dict[str, Any]]:
        action_map = []
        # 1. Move actions (NSEW, 1 cell at a time)
        action_map.append({"type": "move", "delta": (0, -1), "name": "move_N"})
        action_map.append({"type": "move", "delta": (0, 1), "name": "move_S"})
        action_map.append({"type": "move", "delta": (1, 0), "name": "move_E"})
        action_map.append({"type": "move", "delta": (-1, 0), "name": "move_W"})

        # 2. Attack actions - specific to the creature
        for i, attack_details in enumerate(creature.attacks):
            action_map.append({"type": "attack", "index": i, "name": f"attack_{attack_details['name']}"})

        # 3. Bonus actions - specific to the creature
        for i, ba_name_or_dict in enumerate(creature.bonus_actions):
            if isinstance(ba_name_or_dict, str):
                action_map.append({"type": "bonus_action", "name": ba_name_or_dict, "index": i})
            elif isinstance(ba_name_or_dict, dict):
                action_map.append({
                    "type": "bonus_action",
                    "name": ba_name_or_dict.get("name", f"bonus_action_{i}"),
                    "index": i,
                    **ba_name_or_dict
                })

        # 4. Standard D&D actions available to every creature
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

    def decode_action(self, actor_idx: int, action_int: int, lenient: bool = False) -> Dict[str, Any]:
        """Turn an action id into a decoded action dict for `actor_idx`.

        lenient=True (used for opponent models that may have been trained
        against a different action space) turns out-of-range ids into a
        harmless 'pass' instead of raising.
        """
        action_map = self.action_maps[actor_idx]
        actor = self.creatures[actor_idx]

        if not (0 <= action_int < len(action_map)):
            if lenient:
                return {"type": "pass", "name": f"pass_invalid_action_id_{action_int}"}
            raise ValueError(
                f"Invalid action integer: {action_int} for actor '{self.labels[actor_idx]}' "
                f"(action map size {len(action_map)}; attacks: {len(actor.attacks)}, "
                f"bonus_actions: {len(actor.bonus_actions)}).")

        action_details = action_map[action_int].copy()

        if action_details["type"] == "attack":
            attack_idx = action_details.get("index")
            if attack_idx is None or not (0 <= attack_idx < len(actor.attacks)):
                return {"type": "pass", "name": f"pass_invalid_attack_idx_{attack_idx}"}
            action_details["name"] = f"attack_{actor.attacks[attack_idx].get('name', f'idx{attack_idx}')}"

        elif action_details["type"] == "bonus_action":
            bonus_idx = action_details.get("index")
            if bonus_idx is None or not (0 <= bonus_idx < len(actor.bonus_actions)):
                return {"type": "pass", "name": f"pass_invalid_bonus_idx_{bonus_idx}"}
            action_details["name"] = f"bonus_{actor.bonus_actions[bonus_idx]}"

        return action_details

    def action_mask(self, actor_idx: int) -> np.ndarray:
        """Boolean mask of which actions are currently legal for `actor_idx`.
        Mirrors the validity rules in execute(); end_turn is always legal so
        the mask is never all-False."""
        actor = self.creatures[actor_idx]
        opponent = self.creatures[self.opponent_of(actor_idx)]
        action_map = self.action_maps[actor_idx]

        distance = self.distance(actor_idx, self.opponent_of(actor_idx))
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
                         and self.is_valid_position(target_x, target_y)
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

    # --- Turn management --------------------------------------------------

    def begin_turn(self, actor_idx: int) -> None:
        self.creatures[actor_idx].start_new_turn()

    def has_bonus_actions(self, creature: Creature) -> bool:
        return bool(creature.bonus_actions) or "nimble_escape" in creature.traits

    def turn_economy_exhausted(self, creature: Creature) -> bool:
        """True when the creature has nothing meaningful left to do this turn."""
        bonus_done = creature.used_bonus_action_this_turn or not self.has_bonus_actions(creature)
        return (creature.used_action_this_turn and bonus_done and
                creature.speed_remaining <= 0 and creature.attacks_remaining <= 0)

    def micro_step_limit(self, creature: Creature) -> int:
        # Enough for a full Dash turn (2x speed in single-cell moves) plus an
        # action, a bonus action, and some slack. Bounds turn length so a
        # policy repeating invalid actions cannot stall the round forever.
        return 2 * creature.speed_total + 3

    # --- Mechanics helpers ------------------------------------------------

    def first_melee_attack_index(self, creature: Creature) -> int:
        for i, attack in enumerate(creature.attacks):
            if attack.get("attack_type", "melee") == "melee":
                return i
        return -1

    def _has_keen_senses(self, creature: Creature) -> bool:
        return any(t.startswith("keen") for t in creature.traits)

    def _attempt_hide(self, actor: Creature, target: Creature,
                      notes: Dict[str, str], label: str) -> bool:
        """Roll Stealth against the target's passive Perception.

        Terrain does not exist yet, so hiding is abstracted: it may only be
        attempted while not adjacent to the enemy (checked by callers), and it
        shifts dice odds rather than concealing the hider's position.
        """
        stealth_mod = actor.skill_modifier("stealth", "dex")
        roll_result = self.dice_roller("1d20") + stealth_mod
        if roll_result >= target.passive_perception:
            actor.conditions.add("hidden")
            notes[f'{label}_hide_outcome'] = (
                f"Hidden (Stealth {roll_result} vs passive Perception {target.passive_perception}).")
            return True
        notes[f'{label}_hide_outcome'] = (
            f"Failed to hide (Stealth {roll_result} vs passive Perception {target.passive_perception}).")
        return False

    def contested_check(self, initiator_mod: int, defender_mod: int) -> bool:
        """Contested ability check; the initiator rolls first. Ties go to the
        defender (the situation stays as it is, per the 5e contest rule)."""
        initiator_roll = self.dice_roller("1d20") + initiator_mod
        defender_roll = self.dice_roller("1d20") + defender_mod
        return initiator_roll > defender_roll

    def best_escape_modifier(self, creature: Creature) -> int:
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
        if self.is_valid_position(new_x, new_y):
            target.position[0] = new_x
            target.position[1] = new_y
            return True
        return False

    def _update_proximity_conditions(self, notes: Dict[str, str]) -> None:
        """Distance-dependent state, re-checked after any position change:
        being adjacent reveals hidden creatures; a grapple ends when the
        grappler is no longer within reach (moving away releases it)."""
        for i, (creature, label) in enumerate(zip(self.creatures, self.labels)):
            distance = self.distance(i, self.opponent_of(i))
            if distance <= 1:
                if creature.has_condition("hidden"):
                    creature.conditions.discard("hidden")
                    notes[f'{label}_hidden_revealed'] = "Revealed by adjacency."
            else:
                if creature.has_condition("grappled"):
                    creature.conditions.discard("grappled")
                    notes[f'{label}_grapple_released'] = "Grapple ended (grappler out of reach)."

    def resolve_attack_modifiers(self, attacker: Creature, defender: Creature,
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

        # Attacker is hidden (unseen attacker)
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

    def _apply_on_hit_effects(self, attack_stats: Dict[str, Any], target: Creature,
                              notes: Dict[str, str], label: str, other_label: str) -> None:
        """Resolve riders like the wolf bite's 'DC 11 STR save or prone'."""
        for effect in attack_stats.get("on_hit_effects", []):
            save_spec = effect.get("save")
            if not save_spec or not target.is_alive:
                continue
            ability = save_spec["ability"]
            dc = save_spec["dc"]
            if target.saving_throw(ability, dc, self.dice_roller):
                notes[f'{label}_rider_outcome'] = (
                    f"{other_label.capitalize()} succeeded on DC {dc} {ability.upper()} save.")
            else:
                condition = effect.get("on_fail", {}).get("condition")
                if condition:
                    target.conditions.add(condition)
                notes[f'{label}_rider_outcome'] = (
                    f"{other_label.capitalize()} failed DC {dc} {ability.upper()} save"
                    + (f" and is {condition}." if condition else "."))

    # --- Execution --------------------------------------------------------

    def execute(self, actor_idx: int, decoded_action: Dict[str, Any]) -> ActionResult:
        """Resolve a single micro-action for the creature at `actor_idx`.

        Mutates creature state and reports what happened. Every combatant —
        learner, scripted opponent, model opponent, spectator-driven — goes
        through this one method, so all sides play by identical rules.
        """
        target_idx = self.opponent_of(actor_idx)
        actor = self.creatures[actor_idx]
        target = self.creatures[target_idx]
        label = self.labels[actor_idx]
        other_label = self.labels[target_idx]

        result = ActionResult()
        notes = result.notes
        action_type = decoded_action["type"]

        if action_type == "move":
            dx, dy = decoded_action["delta"]
            # Crawling while prone costs 1 extra cell of movement per cell.
            cost = (abs(dx) + abs(dy)) * (2 if actor.has_condition("prone") else 1)
            if actor.has_condition("grappled"):
                notes[f'{label}_move_outcome'] = "Cannot move while grappled."
            elif actor.speed_remaining < cost:
                notes[f'{label}_move_outcome'] = "Not enough speed."
            else:
                target_x = actor.position[0] + dx
                target_y = actor.position[1] + dy
                if not self.is_valid_position(target_x, target_y):
                    notes[f'{label}_move_outcome'] = "Move out of bounds."
                elif [target_x, target_y] == target.position and target.is_alive:
                    notes[f'{label}_move_outcome'] = f"Blocked by {other_label}."
                else:
                    # The move is definitely happening; resolve an opportunity
                    # attack first if the actor is leaving the target's reach.
                    if target.can_act() and \
                       self.distance(actor_idx, target_idx) == 1 and \
                       self.distance_between([target_x, target_y], target.position) > 1 and \
                       not actor.is_disengaging and \
                       not target.used_reaction_this_round:
                        melee_idx = self.first_melee_attack_index(target)
                        if melee_idx != -1:
                            notes[f'opportunity_attack_by_{other_label}'] = (
                                f"{other_label.capitalize()} opportunity attack from {target.position} "
                                f"as {label} moves from {actor.position} to {[target_x, target_y]}")
                            oa_stats = target.attacks[melee_idx]
                            oa_status, oa_messages = self.resolve_attack_modifiers(
                                target, actor, oa_stats, 1, other_label, label)
                            if oa_messages:
                                notes[f'opportunity_attack_by_{other_label}_modifier'] = " ".join(oa_messages)
                            hp_before = actor.current_hp
                            hit, _ = target.make_attack(actor, melee_idx, self.dice_roller,
                                                        advantage_disadvantage=oa_status)
                            target.used_reaction_this_round = True
                            if target.has_condition("hidden"):
                                target.conditions.discard("hidden")
                                notes[f'{other_label}_hidden_revealed'] = "Revealed by attacking."
                            damage = hp_before - actor.current_hp
                            if hit:
                                result.damage.append((target_idx, actor_idx, damage))
                                notes[f'opportunity_attack_by_{other_label}_outcome'] = f"Hit, dealt {damage} damage."
                            else:
                                notes[f'opportunity_attack_by_{other_label}_outcome'] = "Miss."
                            if not actor.is_alive:
                                result.deaths.append(actor_idx)
                                notes['combat_outcome'] = f"{label.capitalize()} died to Opportunity Attack"
                                return result
                    if actor.move(dx, dy, self.map_width, self.map_height, cost=cost):
                        result.success = True
                        notes[f'{label}_move_outcome'] = f"Moved ({dx},{dy})."
                        self._update_proximity_conditions(notes)
                    else:
                        notes[f'{label}_move_outcome'] = "Move failed unexpectedly."

        elif action_type == "attack":
            attack_idx = decoded_action["index"]
            attack_stats = actor.attacks[attack_idx]
            attack_kind = attack_stats.get("attack_type", "melee")

            if actor.used_action_this_turn and actor.attacks_remaining <= 0:
                notes[f'{label}_action_details'] = f"{label.capitalize()} tried Attack but action already used."
                notes[f'{label}_attack_outcome'] = "Action already used."
            elif (actor.used_action_this_turn and actor.attack_type_lock is not None
                  and attack_kind != actor.attack_type_lock):
                notes[f'{label}_action_details'] = (
                    f"{label.capitalize()} Multiattack is locked to {actor.attack_type_lock} attacks this turn.")
                notes[f'{label}_attack_outcome'] = "Wrong attack type for Multiattack."
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
                distance = self.distance(actor_idx, target_idx)

                adv_disadv, modifier_messages = self.resolve_attack_modifiers(
                    actor, target, attack_stats, distance, label, other_label)
                if modifier_messages:
                    notes[f'{label}_attack_modifier'] = " ".join(modifier_messages)

                notes[f'{label}_action_details'] = (
                    f"Attack: {attack_name} (Range: {attack_range}/{max_range}, Type: {attack_kind}, "
                    f"Adv/Disadv: {adv_disadv}), Target Dist: {distance}, Attacks left: {actor.attacks_remaining}")

                if distance <= max_range:
                    hp_before = target.current_hp
                    hit, _ = actor.make_attack(target, attack_idx, self.dice_roller,
                                               advantage_disadvantage=adv_disadv)
                    if actor.has_condition("hidden"):
                        # Attacking reveals you, hit or miss (advantage already applied)
                        actor.conditions.discard("hidden")
                        notes[f'{label}_hidden_revealed'] = "Revealed by attacking."
                    damage = hp_before - target.current_hp
                    if hit:
                        result.damage.append((actor_idx, target_idx, damage))
                        notes[f'{label}_attack_outcome'] = f"Hit, dealt {damage} damage."
                    else:
                        notes[f'{label}_attack_outcome'] = "Miss."
                    if not target.is_alive:
                        result.deaths.append(target_idx)
                        notes['combat_outcome'] = f"{label.capitalize()} won"
                    elif hit:
                        self._apply_on_hit_effects(attack_stats, target, notes, label, other_label)
                else:
                    notes[f'{label}_attack_outcome'] = "Out of range."
                result.success = True

        elif action_type == "bonus_action":
            if actor.used_bonus_action_this_turn:
                notes[f'{label}_action_details'] = f"{label.capitalize()} tried Bonus Action but it was already used."
                notes[f'{label}_bonus_outcome'] = "Bonus action already used."
            else:
                ba_name = decoded_action["name"]
                notes[f'{label}_action_details'] = f"Bonus Action: {ba_name}"
                if ba_name == "bonus_move_1_cell":
                    actor.speed_remaining += 1
                    notes[f'{label}_bonus_outcome'] = "Used bonus_move_1_cell, gained 1 speed."
                else:
                    notes[f'{label}_bonus_outcome'] = f"Used bonus action: {ba_name} (effect placeholder)."
                actor.used_bonus_action_this_turn = True
                result.success = True

        elif action_type in ("dash", "disengage", "dodge"):
            if actor.used_action_this_turn:
                notes[f'{label}_action_details'] = (f"{label.capitalize()} tried {action_type.capitalize()} "
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
                result.success = True
                notes[f'{label}_action_details'] = f"{label.capitalize()} used {action_type.capitalize()} action."

        elif action_type == "stand_up":
            if not actor.has_condition("prone"):
                notes[f'{label}_action_details'] = f"{label.capitalize()} tried to stand up but is not prone."
            else:
                cost = max(1, actor.speed_total // 2)  # Standing costs half your speed
                if actor.speed_remaining >= cost:
                    actor.speed_remaining -= cost
                    actor.conditions.discard("prone")
                    result.success = True
                    notes[f'{label}_action_details'] = f"{label.capitalize()} stood up (cost {cost} movement)."
                else:
                    notes[f'{label}_action_details'] = (
                        f"{label.capitalize()} cannot stand up (needs {cost} movement, "
                        f"has {actor.speed_remaining}).")

        elif action_type == "hide":
            distance = self.distance(actor_idx, target_idx)
            if actor.used_action_this_turn:
                notes[f'{label}_action_details'] = f"{label.capitalize()} tried Hide but action already used."
            elif actor.has_condition("hidden"):
                notes[f'{label}_action_details'] = f"{label.capitalize()} is already hidden."
            elif distance <= 1:
                notes[f'{label}_action_details'] = (
                    f"{label.capitalize()} cannot hide while adjacent to the {other_label}.")
            else:
                actor.used_action_this_turn = True
                result.success = True  # the action is legal; the Stealth roll may still fail
                self._attempt_hide(actor, target, notes, label)
                notes[f'{label}_action_details'] = f"{label.capitalize()} used Hide action."

        elif action_type == "search":
            if actor.used_action_this_turn:
                notes[f'{label}_action_details'] = f"{label.capitalize()} tried Search but action already used."
            elif not target.has_condition("hidden"):
                notes[f'{label}_action_details'] = f"{label.capitalize()} searched but nothing is hidden."
            else:
                actor.used_action_this_turn = True
                result.success = True
                perception_mod = actor.skill_modifier("perception", "wis")
                # Keen senses (hearing/sight/smell) grant advantage on the check
                roll_str = "2d20kh1" if self._has_keen_senses(actor) else "1d20"
                roll_result = self.dice_roller(roll_str) + perception_mod
                dc = 10 + target.skill_modifier("stealth", "dex")
                if roll_result >= dc:
                    target.conditions.discard("hidden")
                    notes[f'{label}_search_outcome'] = (
                        f"Found the {other_label} (Perception {roll_result} vs DC {dc}).")
                else:
                    notes[f'{label}_search_outcome'] = (
                        f"Search failed (Perception {roll_result} vs DC {dc}).")
                notes[f'{label}_action_details'] = f"{label.capitalize()} used Search action."

        elif action_type in ("shove_prone", "shove_push", "grapple"):
            verb = "Grapple" if action_type == "grapple" else "Shove"
            distance = self.distance(actor_idx, target_idx)
            if distance > 1:
                notes[f'{label}_action_details'] = (
                    f"{label.capitalize()} tried {verb} but the {other_label} is out of reach.")
            elif actor.used_action_this_turn and actor.attacks_remaining <= 0:
                notes[f'{label}_action_details'] = f"{label.capitalize()} tried {verb} but action already used."
            else:
                # A special melee attack: replaces one attack of the Attack action.
                if not actor.used_action_this_turn:
                    actor.used_action_this_turn = True
                    actor.attacks_remaining = max((a.get("num_attacks", 1) for a in actor.attacks), default=1)
                actor.attacks_remaining -= 1
                result.success = True
                if actor.has_condition("hidden"):
                    actor.conditions.discard("hidden")
                    notes[f'{label}_hidden_revealed'] = "Revealed by attacking."

                attacker_mod = actor.skill_modifier("athletics", "str")
                defender_mod = self.best_escape_modifier(target)
                notes[f'{label}_action_details'] = (
                    f"{label.capitalize()} attempted {verb} "
                    f"(Athletics {attacker_mod:+d} vs best of Athletics/Acrobatics {defender_mod:+d}).")
                if self.contested_check(attacker_mod, defender_mod):
                    if action_type == "grapple":
                        target.conditions.add("grappled")
                        notes[f'{label}_contest_outcome'] = f"Grapple succeeded: {other_label} is grappled."
                    elif action_type == "shove_prone":
                        target.conditions.add("prone")
                        notes[f'{label}_contest_outcome'] = f"Shove succeeded: {other_label} is prone."
                    else:  # shove_push
                        if self._push_away(actor, target):
                            notes[f'{label}_contest_outcome'] = (
                                f"Shove succeeded: {other_label} pushed to {target.position}.")
                            self._update_proximity_conditions(notes)
                        else:
                            notes[f'{label}_contest_outcome'] = (
                                f"Shove succeeded but the {other_label} has nowhere to go.")
                else:
                    notes[f'{label}_contest_outcome'] = f"{verb} failed (contest lost)."

        elif action_type == "escape_grapple":
            if not actor.has_condition("grappled"):
                notes[f'{label}_action_details'] = f"{label.capitalize()} tried to escape but is not grappled."
            elif actor.used_action_this_turn:
                notes[f'{label}_action_details'] = f"{label.capitalize()} tried to escape but action already used."
            else:
                actor.used_action_this_turn = True
                result.success = True
                escaper_mod = self.best_escape_modifier(actor)
                grappler_mod = target.skill_modifier("athletics", "str")
                notes[f'{label}_action_details'] = f"{label.capitalize()} tried to escape the grapple."
                if self.contested_check(escaper_mod, grappler_mod):
                    actor.conditions.discard("grappled")
                    notes[f'{label}_contest_outcome'] = "Escaped the grapple."
                else:
                    notes[f'{label}_contest_outcome'] = "Failed to escape the grapple."

        elif action_type in ("bonus_disengage", "bonus_hide"):
            distance = self.distance(actor_idx, target_idx)
            if "nimble_escape" not in actor.traits:
                notes[f'{label}_action_details'] = f"{label.capitalize()} lacks Nimble Escape."
            elif actor.used_bonus_action_this_turn:
                notes[f'{label}_action_details'] = (
                    f"{label.capitalize()} tried Bonus Action but it was already used.")
            elif action_type == "bonus_hide" and actor.has_condition("hidden"):
                notes[f'{label}_action_details'] = f"{label.capitalize()} is already hidden."
            elif action_type == "bonus_hide" and distance <= 1:
                notes[f'{label}_action_details'] = (
                    f"{label.capitalize()} cannot hide while adjacent to the {other_label}.")
            else:
                actor.used_bonus_action_this_turn = True
                result.success = True
                if action_type == "bonus_disengage":
                    actor.is_disengaging = True
                    notes[f'{label}_action_details'] = (
                        f"{label.capitalize()} used Nimble Escape: Disengage (bonus action).")
                else:
                    self._attempt_hide(actor, target, notes, label)
                    notes[f'{label}_action_details'] = (
                        f"{label.capitalize()} used Nimble Escape: Hide (bonus action).")

        elif action_type == "end_turn":
            result.success = True
            notes[f'{label}_action_details'] = f"{label.capitalize()} ended turn."

        elif action_type == "pass":
            # Fallback from decode_action for an index this creature doesn't have.
            notes[f'{label}_action_details'] = (f"{label.capitalize()} chose an action invalid for this "
                                                f"creature; treated as pass.")

        return result
