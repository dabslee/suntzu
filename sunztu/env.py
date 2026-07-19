from typing import List, Dict, Callable, Tuple, Optional, Any, Union
from collections import deque
import random
import re
import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from gymnasium.wrappers import FlattenObservation
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    class FlattenObservation: pass

from sunztu.dice import roll
from sunztu.creature import Creature, DEFAULT_ABILITIES
from sunztu.models import load_model_any
from sunztu.opponents import ModelOpponent, RuleBasedOpponent
from sunztu.rules import CombatRules


class DnDCombatEnv(gym.Env):
    metadata = {'render_modes': ['ansi', 'human', 'rgb_array'], 'render_fps': 4}

    # The learner always occupies index 0 in the rules engine ('agent' seat);
    # the opponent is index 1. Rewards are computed relative to this index.
    LEARNER_IDX = 0

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
            
        # Play-by-play log of what each side did, shown in the render panel
        # and inspectable by runners (e.g. run_env.py prints new entries).
        self.event_log: deque = deque(maxlen=200)

        # Rendering is delegated to sunztu.render (pygame lives only there).
        # Created eagerly for windowed modes so a missing pygame fails fast.
        self._renderer = None
        if self.render_mode in ('human', 'rgb_array'):
            try:
                from sunztu.render import PygameRenderer
            except ImportError as e:
                raise ImportError(
                    "Pygame is required for 'human'/'rgb_array' render modes "
                    "(pip install pygame).") from e
            self._renderer = PygameRenderer(self, self.render_mode)


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
        
        # The rules engine owns the creatures and all combat mechanics; this
        # env is the RL adapter on top of it (rewards, observations, learner
        # binding, opponent turn orchestration). The labels 'agent'/'enemy'
        # are display-only — the engine itself is index-based.
        self.rules = CombatRules([self.agent, self.enemy], ["agent", "enemy"],
                                 self.map_width, self.map_height, dice_roller=roll)
        self.action_space = spaces.Discrete(len(self.rules.action_maps[0]))

        # The opponent seat's policy: a loaded model if one was given,
        # otherwise the scripted rule-based AI.
        if self.enemy_is_model_controlled and self.enemy_model and self.enemy_obs_flattener:
            self.enemy_policy = ModelOpponent(
                self.enemy_model,
                lambda: self.enemy_obs_flattener.observation(self._get_enemy_observation()))
        else:
            self.enemy_policy = RuleBasedOpponent()
        
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

    @property
    def dice_roller(self):
        return self.rules.dice_roller

    @dice_roller.setter
    def dice_roller(self, fn) -> None:
        # Tests inject scripted dice by assigning env.dice_roller; the engine
        # is the single owner, this property just forwards.
        self.rules.dice_roller = fn

    def _calculate_manhattan_distance(self, pos1: List[int], pos2: List[int]) -> int:
        return self.rules.distance_between(pos1, pos2)

    def _is_valid_position(self, x: int, y: int) -> bool:
        return self.rules.is_valid_position(x, y)

    @property
    def action_map(self):
        """The learner's action map (index 0 in the rules engine)."""
        return self.rules.action_maps[0]

    def _decode_action(self, action_int: int, is_enemy: bool = False) -> Dict[str, Any]:
        # Decode against the acting creature's own map (the two sides may have
        # different attacks/bonus actions/traits). Lenient for the enemy seat:
        # an opponent model trained against a different action space may
        # predict ids this creature doesn't have.
        return self.rules.decode_action(1 if is_enemy else 0, action_int, lenient=is_enemy)

    def _turn_economy_exhausted(self, creature: Creature) -> bool:
        return self.rules.turn_economy_exhausted(creature)

    def _max_micro_steps(self, creature: Creature) -> int:
        return self.rules.micro_step_limit(creature)

    def action_masks(self) -> np.ndarray:
        """sb3-contrib MaskablePPO hook: validity mask over the learner's actions."""
        return self.rules.action_mask(0)

    # --- Spectator API: drive either combatant one micro-action at a time ---
    #
    # step() bundles one agent micro-action with the opponent's *entire* turn,
    # which is right for RL training but means a viewer can only render one
    # frame for the whole opponent turn. These helpers let run_env drive both
    # sides symmetrically (one micro-action, one frame) without the training
    # asymmetry. 'prefix' is 'agent' (Agent 1 seat) or 'enemy' (Agent 2 seat).

    def _side(self, prefix: str) -> Tuple[Creature, Creature, List[Dict[str, Any]]]:
        if prefix == 'agent':
            return self.agent, self.enemy, self.rules.action_maps[0]
        return self.enemy, self.agent, self.rules.action_maps[1]

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
        return self.rules.action_mask(0 if prefix == 'agent' else 1)

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

    def _resolve_attack_modifiers(self, attacker: Creature, defender: Creature,
                                  attack_stats: Dict[str, Any], distance: int,
                                  attacker_label: str, defender_label: str) -> Tuple[Optional[str], List[str]]:
        return self.rules.resolve_attack_modifiers(
            attacker, defender, attack_stats, distance, attacker_label, defender_label)

    def _execute_micro_action(self, actor: Creature, target: Creature,
                              decoded_action: Dict[str, Any], info: Dict[str, Any],
                              prefix: str) -> Tuple[float, bool, bool]:
        """Adapter over CombatRules.execute(): resolve the action through the
        rules engine, merge its notes into `info`, and translate what happened
        into the learner's reward.

        The reward policy lives HERE, not in the engine: hp the opponent loses
        is +reward, hp the learner loses is -reward, and a death ends the
        episode at +/-100 depending on who fell. (`actor`/`target` are kept in
        the signature for callers and tests; the seat is derived from
        `prefix`.)
        """
        actor_idx = 0 if prefix == 'agent' else 1
        result = self.rules.execute(actor_idx, decoded_action)
        info.update(result.notes)

        reward_delta = 0.0
        for _attacker_idx, victim_idx, damage in result.damage:
            reward_delta += -damage if victim_idx == self.LEARNER_IDX else damage
        for dead_idx in result.deaths:
            reward_delta += -100 if dead_idx == self.LEARNER_IDX else 100

        terminated = bool(result.deaths)
        return reward_delta, terminated, result.success

    def _run_enemy_turn(self, info: Dict[str, Any]) -> Tuple[float, bool]:
        """Run the opponent's complete turn, one policy-chosen micro-action at
        a time. The policy (rule-based or model) only chooses; execution and
        logging go through the exact same path for both. Returns
        (reward_delta, terminated)."""
        self.rules.begin_turn(1)
        info['enemy_ai_type'] = self.enemy_policy.kind
        reward_delta = 0.0
        terminated = False
        for _ in range(self.rules.micro_step_limit(self.enemy)):
            decoded = self.enemy_policy.choose(self.rules, 1)
            if decoded is None:
                break  # the policy has nothing more to do this turn

            # Fresh outcome dict so each micro-action logs on its own line
            # (multiattack and multi-cell moves would overwrite each other).
            sub: Dict[str, Any] = {}
            r, terminated, success = self._execute_micro_action(
                self.enemy, self.agent, decoded, sub, 'enemy')
            info.update(sub)
            self._log_micro_action('enemy', decoded, sub)
            reward_delta += r

            if terminated or decoded["type"] == "end_turn" or self._turn_economy_exhausted(self.enemy):
                break
            if not success:
                # A failed action leaves the state unchanged, so a
                # deterministic policy would repeat it forever; end the turn.
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

    def render(self) -> Optional[Union[np.ndarray, str]]:
        if self.render_mode is None:
            gym.logger.warn("render() called without a render mode; defaulting to 'ansi'.")
            self.render_mode = 'ansi'

        if self.render_mode in ('human', 'rgb_array'):
            return self._renderer.render()

        if self.render_mode == 'ansi':
            grid = [['.' for _ in range(self.map_width)] for _ in range(self.map_height)]
            if self.enemy.is_alive and self._is_valid_position(*self.enemy.position):
                grid[self.enemy.position[1]][self.enemy.position[0]] = 'E'
            if self.agent.is_alive and self._is_valid_position(*self.agent.position):
                grid[self.agent.position[1]][self.agent.position[0]] = 'A'
            ansi_str = "\n".join(" ".join(row) for row in grid)
            ansi_str += (f"\nAgent HP: {self.agent.current_hp}/{self.agent.max_hp}, "
                         f"Speed: {self.agent.speed_remaining}/{self.agent.speed_total} "
                         f"| Enemy HP: {self.enemy.current_hp}/{self.enemy.max_hp}")
            print(ansi_str)
            return ansi_str

        raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
