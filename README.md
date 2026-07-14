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
| [dnd_gym_env.py](dnd_gym_env.py) | The core module: `Creature` (stats + combat state) and `DnDCombatEnv` (the Gymnasium environment, including pygame rendering and optional model-controlled enemy). |
| [dice.py](dice.py) | Dice-string roller, e.g. `1d8+3`, `2d20kh1` (advantage), `2d20kl1` (disadvantage). |
| [bestiary/](bestiary/) | Creature stat blocks (`commoner`, `wolf`, `cat`, `bandit`, `guard`, `goblin`, `scout`). Each module exposes `get_<name>_stats()` returning a dict of AC, HP, speed, attacks, and bonus actions. |
| [train_agent.py](train_agent.py) | Trains a single DQN agent (guard vs. wolf by default) against the rule-based enemy or a saved enemy model, then runs a 20-episode evaluation. |
| [train_alternating.py](train_alternating.py) | Alternating self-play: trains model A against the latest model B, then B against the updated A, for N iterations. |
| [evaluate_models.py](evaluate_models.py) | Pits two saved models against each other for many episodes and reports win/loss/draw rates. |
| [run_env.py](run_env.py) | Interactive/visual runner with pygame rendering; agent can be random or model-controlled; can export PNG frames per step. |
| [test_dice.py](test_dice.py), [test_dnd_gym_env.py](test_dnd_gym_env.py) | Pytest suites for the dice roller and the environment. |
| `models/` | Saved `.zip` DQN checkpoints from past runs. **Note:** checkpoints predating the turn-structure fix (see below) were trained on broken dynamics and a different action-space size; they should be retrained. |
| `dnd_dqn_tensorboard/`, `dnd_selfplay_tensorboard/` | TensorBoard logs from past runs. |
| `run_env_output/` | Exported PNG frames from `run_env.py` runs. |

## How the environment works

`DnDCombatEnv` is a standard `gym.Env` with a `Dict` observation space and a
`Discrete` action space. Training scripts wrap it in
`gymnasium.wrappers.FlattenObservation` (SB3's `MlpPolicy` needs a flat vector)
and then `DummyVecEnv`.

### Observation space

All values are from the agent's perspective:

- `agent_hp_norm`, `enemy_hp_norm` — HP as a fraction of max HP
- `agent_pos`, `enemy_pos` — grid coordinates
- `distance_to_enemy` — Manhattan distance
- `agent_speed_remaining_norm` — remaining movement this turn (clamped to 1.0)
- `agent_used_action`, `agent_used_bonus_action`, `agent_used_reaction`,
  `agent_is_dodging` — action-economy flags (0/1)

### Action space

Built per-creature by `_build_action_map_for_creature`:

1. Four single-cell moves (N/S/E/W)
2. One action per attack in the creature's stat block
3. One action per bonus action in the stat block
4. Dash, Disengage, Dodge
5. `end_turn` (hand the round over to the opponent)

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

Implemented 5e mechanics: attack rolls vs. AC with advantage/disadvantage,
ranged attacks at disadvantage when adjacent, the Dodge action imposing
disadvantage, Dash adding speed, Disengage suppressing opportunity attacks, and
opportunity attacks when a creature leaves melee reach.

### Rewards (per step, from the agent's perspective)

- `-0.1` flat step penalty
- `+damage` dealt to the enemy / `-damage` taken from the enemy (actual HP
  lost, including damage dealt or taken via opportunity attacks)
- `+100` for killing the enemy, `-100` for dying
- `+0.05` for Dash/Disengage/Dodge, `+0.02` for the example bonus action
- `-0.05` for choosing an unavailable action (e.g. attacking when the action
  was already used, or moving into a wall)

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

# Evaluate two saved models head-to-head
python evaluate_models.py --model1_path models/commoner_v_wolf_3.zip --model2_path models/wolf_v_commoner_3.zip \
    --agent_stats_module bestiary.commoner --enemy_stats_module bestiary.wolf --num_episodes 1000

# Watch an episode with pygame rendering (random or model-controlled agent)
python run_env.py --agent_control model --agent_model_path dnd_dqn_agent.zip

# Run tests
pytest test_dice.py test_dnd_gym_env.py
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

- `num_attacks` (Multiattack) is parsed but not used — every attack action is
  a single attack roll.
- Special monster traits (Pack Tactics, Nimble Escape, Hide) are not
  implemented; goblin bonus actions are placeholders with no effect.
- No critical hits/misses (a natural 20/1 is just a high/low roll), no
  diagonal movement, no terrain or obstacles.
- The rule-based enemy AI is simple: close to range of its first attack, then
  attack. It never uses Dash, Dodge, Disengage, or ranged/melee switching.
