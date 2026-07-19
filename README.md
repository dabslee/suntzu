# sunztu

An experiment in training reinforcement-learning agents to fight D&D 5e-style
grid combat, built on [Gymnasium](https://gymnasium.farama.org/) (the maintained
fork of OpenAI Gym) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
DQN.

Two creatures face off on a small square grid. One (the "agent") is controlled
by the RL policy being trained; the other (the "enemy") is controlled either by
a simple rule-based AI or by a previously trained model, which enables
alternating self-play.

## Repository layout

| File / dir | Purpose |
|---|---|
| [sunztu/env.py](sunztu/env.py) | `DnDCombatEnv` — the Gymnasium environment: turn structure, combat rules executor, observations, action masks, opponent AIs, and the spectator API. |
| [sunztu/creature.py](sunztu/creature.py) | The `Creature` data model: stats, action economy, conditions, attack rolls, saving throws. |
| [sunztu/dice.py](sunztu/dice.py) | Dice-string roller, e.g. `1d8+3`, `2d20kh1` (advantage), `2d20kl1` (disadvantage). |
| [sunztu/render.py](sunztu/render.py) | All pygame rendering (tokens, badges, stats/log panel). Only imported for `human`/`rgb_array` modes, so headless training has no pygame dependency. |
| [sunztu/models.py](sunztu/models.py) | Algorithm-agnostic checkpoint loading (`load_model_any` handles MaskablePPO and legacy DQN zips). |
| [sunztu/spectator.py](sunztu/spectator.py) | Shared symmetric driving loop used by `run_env.py` and `evaluate_models.py`, so what you watch is exactly what gets scored. |
| [sunztu/training.py](sunztu/training.py) | Shared trainer setup: vectorized env stack (FlattenObservation → DummyVecEnv → VecNormalize) and default PPO hyperparameters. |
| [bestiary/](bestiary/) | Creature stat blocks (`commoner`, `wolf`, `cat`, `bandit`, `guard`, `goblin`, `scout`) plus token art. Each module exposes `get_<name>_stats()`; the package exposes `get_stats(name)` and `load_stats_from_module(path)`. |
| [train_agent.py](train_agent.py) | Trains a single MaskablePPO agent (guard vs. wolf by default) against the rule-based enemy or a saved enemy model, then runs a 20-episode evaluation. |
| [train_alternating.py](train_alternating.py) | Alternating self-play with MaskablePPO: trains model A against the latest model B, then B against the updated A, for N iterations. |
| [evaluate_models.py](evaluate_models.py) | Scores two policies head-to-head over many episodes, driving both sides symmetrically (same loop as the viewer). Either side may be left model-less to act randomly as a baseline; `--swap_sides` also runs the mirror matchup to cancel out the first-turn advantage. |
| [run_env.py](run_env.py) | Symmetric spectator: drives **both** combatants one micro-action at a time, so every micro-action — including each of the opponent's moves — gets its own frame and log line. Creature-art tokens with name labels and condition badges, framed as **Agent 1** and **Agent 2**. Each side is model-controlled if given a model (`--agent1_model`/`--agent2_model`), else random. Selectable matchup (`--agent1_creature`/`--agent2_creature`), optional per-micro-action PNG export. |
| [tests/](tests/) | Pytest suite, split by area (`test_creature`, `test_env_core`, `test_mechanics`, `test_masks`, `test_dice`) with shared fixtures in `conftest.py`. |
| [dnd_gym_env.py](dnd_gym_env.py) | Deprecated compatibility shim re-exporting the old names from the `sunztu` package. |
| `models/` | Saved `.zip` checkpoints. **Note:** checkpoints predating the turn-structure fix (see below) were trained on broken dynamics and a different action-space size; they should be retrained. |
| `dnd_ppo_tensorboard/`, `dnd_selfplay_tensorboard/` | TensorBoard logs from training runs. |

## How the environment works

`DnDCombatEnv` is a standard `gym.Env` with a `Dict` observation space and a
`Discrete` action space. Training scripts wrap it in
`gymnasium.wrappers.FlattenObservation` (SB3's `MlpPolicy` needs a flat vector)
and then `DummyVecEnv`. The env also exposes `action_masks()` (the sb3-contrib
MaskablePPO hook): a boolean validity mask mirroring the executor's rules, so
a masked policy never samples currently-illegal actions.

### Observation space

All values are from the agent's perspective:

- `agent_hp_norm`, `enemy_hp_norm` — HP as a fraction of max HP
- `agent_pos`, `enemy_pos` — grid coordinates
- `distance_to_enemy` — Manhattan distance
- `agent_speed_remaining_norm` — remaining movement this turn (clamped to 1.0)
- `agent_used_action`, `agent_used_bonus_action`, `agent_used_reaction`,
  `agent_is_dodging` — action-economy flags (0/1)
- `agent_attacks_remaining_norm` — unspent Multiattack rolls this turn
- `agent_is_prone` / `agent_is_hidden` / `agent_is_grappled` and
  `enemy_is_prone` / `enemy_is_hidden` / `enemy_is_grappled` /
  `enemy_is_dodging` — condition flags (hidden/grappled are wired but always 0
  until the stealth and contests phases land)

The same builder produces the model-controlled enemy's observation with roles
swapped, so both seats always see symmetric features.

### Action space

Built per-creature by `_build_action_map_for_creature`:

1. Four single-cell moves (N/S/E/W)
2. One action per attack in the creature's stat block (taking the first attack
   of a turn grants `num_attacks` Multiattack rolls, spent one per step; the
   first attack's melee/ranged type locks the rest, per the scout's "two melee
   attacks or two ranged attacks")
3. One action per bonus action in the stat block
4. Dash, Disengage, Dodge, `stand_up` (from prone; costs half speed), Hide,
   Search, and the special melee attacks `shove_prone`, `shove_push`,
   `grapple`, plus `escape_grapple`
5. Trait-driven bonus actions: creatures with Nimble Escape (goblin) get
   `bonus_disengage` and `bonus_hide`
6. `end_turn` (hand the round over to the opponent)

Because the action space size depends on the creature's attack/bonus-action
count, models trained for one creature are not interchangeable with another
creature's action space.

### Turn model

Each `env.step(action)` executes **one agent micro-action** (a single-cell
move, an attack, a bonus action, Dash/Disengage/Dodge, or `end_turn`). The
agent keeps taking micro-actions across successive steps until its turn ends,
which happens when it:

- chooses `end_turn`,
- exhausts its action economy (action + bonus action used, no movement left), or
- hits a per-turn micro-step cap (`2 × speed + 3`, enough for a full Dash turn
  — this prevents a policy from stalling the round with invalid actions).

Only then does the enemy take its **complete turn** (movement plus attack for
the rule-based AI; a symmetric micro-action loop for a model-controlled
enemy), after which the agent's action economy refreshes and the returned
observation describes the fresh state it will decide from next. Both sides run
through the same shared action executor, so agent and enemy play by identical
rules — important for self-play, where a model trained in the agent seat is
later deployed in the enemy seat.

Implemented 5e mechanics: attack rolls vs. AC with advantage/disadvantage
(all sources are collected and stacked with the real 5e rule — any advantage
plus any disadvantage yields a straight roll), Multiattack, ranged attacks at
disadvantage when adjacent, normal/long weapon ranges (disadvantage past
normal range, invalid past long range), on-hit saving-throw riders (the wolf's
bite knocks its target prone on a failed DC 11 STR save), the prone condition
(advantage for adjacent attackers, disadvantage at range and on the prone
creature's own attacks, crawling at double cost, standing for half speed),
the Dodge action, Dash, Disengage, opportunity attacks when a creature leaves
melee reach, ability scores/saving throws, and per-creature skills and passive
Perception.

**Stealth** (abstracted — there is no terrain, so hiding shifts dice odds
rather than concealing position, and positions stay fully observable): Hide is
allowed only while not adjacent, rolling Stealth vs. the enemy's passive
Perception. While hidden your attacks have advantage and attacks against you
have disadvantage; attacking or becoming adjacent reveals you. Search spends
the Action on an active Perception check vs. `10 + Stealth` (keen-sense traits
grant advantage). The rule-based AI searches when its target is hidden.

**Contests**: Shove (knock prone or push one cell) and Grapple replace one
attack of the Attack action and are resolved as contested checks — Athletics
vs. the defender's best of Athletics/Acrobatics, ties to the defender. A
grappled creature cannot move until it spends its Action to escape (contested
again) or the grappler moves out of reach, which releases the grapple
(dragging is not modeled).

Creature traits are declarative: `nimble_escape` adds the goblin's bonus
actions, `keen_*` grants Search advantage, and `pack_tactics` is wired into
the advantage logic but inert until multi-creature combat exists.

### Rewards (per step, from the agent's perspective)

- `-0.1` flat step penalty
- `+damage` dealt to the enemy / `-damage` taken from the enemy (actual HP
  lost, including damage dealt or taken via opportunity attacks)
- `+100` for killing the enemy, `-100` for dying
- `-0.05` for choosing an unavailable action (e.g. attacking when the action
  was already used, or moving into a wall)

There is deliberately no shaping bonus for utility actions (the old +0.05 for
Dash/Dodge/Disengage was removed when the action space grew) — their value has
to flow through damage avoided and fights won.

A step that ends the agent's turn also includes the reward consequences of the
enemy's entire turn. Episodes truncate at `max_episode_steps` (default 100
micro-steps).

### Enemy control

If `enemy_model_path` is passed, the env loads a saved DQN and, each enemy
turn, runs a micro-action loop: it builds an observation *from the enemy's
perspective* (roles swapped), flattens it with a `FlattenObservation` built
around a role-swapped dummy env, and executes the model's predicted action,
repeating until the model ends its turn, exhausts its economy, or picks an
action that fails (which would loop forever under a deterministic policy).
Otherwise a rule-based AI moves toward the agent until within attack range,
then attacks.

## Usage

```bash
pip install -r requirements.txt

# Train a guard vs. rule-based wolf (50k timesteps), then auto-evaluate
python train_agent.py
# ...or against a saved enemy model
python train_agent.py --enemy_model_path models/wolf_v_commoner_3.zip

# Alternating self-play (wolf vs. commoner, 10 iterations x 50k steps each)
python train_alternating.py --agent1_stats_module bestiary.wolf --agent2_stats_module bestiary.commoner

# Evaluate two saved models head-to-head (add --swap_sides for the mirror matchup)
python evaluate_models.py \
    --agent1_stats_module bestiary.wolf --agent2_stats_module bestiary.goblin \
    --agent1_model models/dnd_agent1_selfplay.zip \
    --agent2_model models/dnd_agent2_selfplay.zip --num_episodes 1000 --swap_sides

# Omit a model to use a random policy on that side as a baseline
python evaluate_models.py \
    --agent1_stats_module bestiary.wolf --agent2_stats_module bestiary.goblin \
    --agent1_model models/dnd_agent1_selfplay.zip --num_episodes 200

# Watch an episode with pygame rendering (both sides driven one action per frame)
python run_env.py --agent1_creature goblin --agent2_creature wolf          # both random
python run_env.py --agent1_creature wolf --agent2_creature goblin \
    --agent1_model models/dnd_agent1_selfplay.zip \
    --agent2_model models/dnd_agent2_selfplay.zip   # trained self-play models fight

# Run tests
pytest
```

TensorBoard: `tensorboard --logdir dnd_dqn_tensorboard` (or
`dnd_selfplay_tensorboard` for self-play runs).

## Fixed training bugs (July 2026)

Training previously never converged. Code inspection plus empirical
verification found the causes, all fixed in the current revision:

1. **The agent's turn never refreshed (critical).** `step()` only reset the
   agent's action economy when it had used no action, no bonus action, *and*
   had full movement — a condition that could never become true again after
   the first move or attack, because only the reset itself restores those.
   Verified empirically: the agent got **one attack per episode** while the
   enemy's economy refreshed every step (a full attack per agent micro-action,
   plus unlimited opportunity attacks). No policy could win. Fixed by the
   explicit turn structure described above.
2. **Asymmetric turn dynamics.** One agent micro-action used to be matched
   against a full enemy turn every step. Now the enemy acts only when the
   agent's turn ends, and both sides share one action executor — so self-play
   models experience the same dynamics in either seat, and enemy
   Dash/Dodge/Disengage actually work.
3. **The `info` dict was discarded.** `step()` built detailed logs (attack
   outcomes, `combat_outcome`, opportunity attacks) but every return statement
   returned a bare `_get_info()` instead. Now merged into the returned info.
4. **Opportunity attacks fired before move validation.** A move into a wall or
   an occupied square could provoke an OA without the mover going anywhere.
   OAs now resolve only when the move is actually happening.
5. **OA rewards used the raw damage roll** (overcounting overkill) instead of
   actual HP lost. All damage rewards now use actual HP deltas.
6. **Duplicate `pass_turn`/`end_turn` actions** consolidated into a single
   `end_turn`. This changed the action-space size, which is one reason old
   checkpoints in `models/` are incompatible (the other being that they were
   trained on the broken dynamics). If a loaded enemy model predicts an action
   id outside the current map, it is treated as a pass rather than crashing.
7. **Hopeless default matchup.** `train_agent.py` trained a commoner (4 HP,
   AC 10) against a wolf — near-unwinnable, so the reward signal was dominated
   by −100 losses. The default is now guard vs. wolf, a winnable fight.

## Remaining limitations

See [ACTION_SPACE_PLAN.md](ACTION_SPACE_PLAN.md) for the full roadmap.
Phases 0–3 are done: abilities/saves/conditions, Multiattack, weapon ranges,
prone, Hide/Search, Nimble Escape, and Shove/Grapple — the complete
no-ally, no-object subset of SRD standard actions for the seven bestiary
creatures. Still missing:

- Pack Tactics and Help need multi-creature combat; declared but inert.
- No critical hits/misses (a natural 20/1 is just a high/low roll), no
  diagonal movement, no terrain or obstacles (so the cat's climb speed is
  meaningless and stealth is abstracted), no Ready action, no spells.
- The rule-based enemy AI is simple: stand up if prone, search if its target
  is hidden, close to range of its first attack, then attack with its full
  Multiattack. It never uses Dash, Dodge, Disengage, shoves, or grapples.

## Training performance

Training uses **MaskablePPO** (sb3-contrib) with three measured ingredients —
switched from single-env DQN after profiling showed >95% of wall time was
per-step PyTorch overhead, not the environment:

1. **8 vectorized envs in one `DummyVecEnv`** — batches the policy forward
   pass (the win is batching, not parallelism; threads/subprocesses don't
   help because the env itself steps at ~18k steps/s while learning ran at
   ~870).
2. **Action masking** via the env's `action_masks()` — with 15–19 actions per
   creature and most invalid at any moment, unmasked exploration wastes most
   of its samples on no-ops.
3. **`VecNormalize(norm_reward=True)`** — PPO's value function chokes on the
   ±100 terminal rewards without it (0% wins even at 150k steps); with it,
   learning takes off. Observations are *not* normalized, so saved policies
   run standalone (as evaluation agents or `enemy_model_path` opponents)
   without needing the normalizer's statistics.

Measured on guard vs. rule-based wolf: DQN managed ~870 steps/s and a 19%
win rate at 30k steps (~35 s); MaskablePPO runs ~3,400 steps/s and reaches
**32% wins in 13 s / 47% wins in 40 s**. Old DQN checkpoints cannot be
*resumed* by the PPO scripts, but they still load fine as opponents and in
`run_env.py`/`evaluate_models.py` (`load_model_any` handles both formats).
