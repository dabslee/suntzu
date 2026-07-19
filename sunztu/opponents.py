"""Opponent policies: how the non-learner side picks its next micro-action.

Both policies expose `choose(rules, actor_idx) -> Optional[decoded_action]`
(None = nothing more to do this turn). The env runs one unified turn loop and
executes whatever the policy picks through the same CombatRules.execute()
path as everything else — the policy only *chooses*, it never resolves.
"""
from typing import Any, Callable, Dict, Optional

import numpy as np

from sunztu.models import uses_action_masks
from sunztu.rules import CombatRules

_MOVE_NAMES = {(0, -1): "move_N", (0, 1): "move_S", (1, 0): "move_E", (-1, 0): "move_W"}


class RuleBasedOpponent:
    """Simple scripted behavior: stand up if prone, search if the target is
    hidden, close to the first attack's range, then attack with every
    Multiattack slot. Stateless — the next action is derived from the current
    combat state each call, using the action mask to skip blocked moves."""

    kind = "rule_based"

    def choose(self, rules: CombatRules, actor_idx: int) -> Optional[Dict[str, Any]]:
        actor = rules.creatures[actor_idx]
        target_idx = rules.opponent_of(actor_idx)
        target = rules.creatures[target_idx]
        attack_stats = actor.attacks[0] if actor.attacks else None
        attack_range = attack_stats.get("range", 1) if attack_stats else 1

        mask = rules.action_mask(actor_idx)
        action_map = rules.action_maps[actor_idx]

        def masked(name: str) -> Optional[Dict[str, Any]]:
            for i, action in enumerate(action_map):
                if action["name"] == name and mask[i]:
                    return dict(action)
            return None

        # 1. Stand up first if prone (costs half speed; crawling costs double).
        #    If it can't afford to stand, fall through and crawl/attack.
        if actor.has_condition("prone"):
            action = masked("stand_up")
            if action:
                return action

        # 2. If the target is hidden, spend the action searching for it rather
        #    than attacking at disadvantage.
        if target.has_condition("hidden") and not actor.used_action_this_turn:
            return {"type": "search", "name": "search"}

        # 3. Close distance until within attack range (or adjacent if unarmed):
        #    prefer the axis with the larger gap, fall back to the other when
        #    the preferred step is blocked; stop when boxed in.
        distance = rules.distance(actor_idx, target_idx)
        if distance > attack_range and actor.speed_remaining > 0:
            diff_x = target.position[0] - actor.position[0]
            diff_y = target.position[1] - actor.position[1]
            step_x = (1 if diff_x > 0 else -1, 0) if diff_x != 0 else None
            step_y = (0, 1 if diff_y > 0 else -1) if diff_y != 0 else None
            ordered = (step_x, step_y) if abs(diff_x) >= abs(diff_y) else (step_y, step_x)
            for delta in ordered:
                if delta is None:
                    continue
                action = masked(_MOVE_NAMES[delta])
                if action:
                    return action

        # 4. Attack while in range, spending every Multiattack slot.
        if (attack_stats and target.is_alive and distance <= attack_range and
                (not actor.used_action_this_turn or actor.attacks_remaining > 0)):
            return {"type": "attack", "index": 0,
                    "name": f"attack_{attack_stats.get('name', '0')}"}

        return None  # nothing useful left to do


class ModelOpponent:
    """A saved policy drives the opponent. `observation_fn` supplies the flat
    observation from the opponent's own perspective (wired by the env, which
    owns the observation schema and the FlattenObservation wrapper)."""

    kind = "model_controlled"

    def __init__(self, model: Any, observation_fn: Callable[[], np.ndarray]):
        self.model = model
        self.observation_fn = observation_fn
        self._masked = uses_action_masks(model)

    def choose(self, rules: CombatRules, actor_idx: int) -> Optional[Dict[str, Any]]:
        flat_obs = self.observation_fn()
        if self._masked:
            action_id, _ = self.model.predict(flat_obs, deterministic=True,
                                              action_masks=rules.action_mask(actor_idx))
        else:
            action_id, _ = self.model.predict(flat_obs, deterministic=True)
        # lenient: a model trained against a different action space may predict
        # an id this creature doesn't have — treat it as a pass, don't crash.
        return rules.decode_action(actor_idx, int(np.asarray(action_id).item()), lenient=True)
