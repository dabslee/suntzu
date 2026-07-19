# Plan: Expanding the Action Space to Full SRD Standard Actions

Goal: make the environment's action space comprehensive enough that the
bandit, cat, commoner, goblin, guard, scout, and wolf can each do everything
their SRD stat block says they can do, plus the standard actions every
creature gets ("Actions in Combat", SRD 5.1). Spells are explicitly out of
scope for now.

Data source: the [D&D 5e SRD API](https://5e-bits.github.io/docs/introduction)
(`https://www.dnd5eapi.co/api/2014/...`). Its monster records are structured —
`actions[].attack_bonus`, `actions[].damage[].damage_dice`, `actions[].dc`,
multiattack composition, `proficiencies`, ability scores, and
`senses.passive_perception` — so stat blocks can be generated rather than
hand-written.

## What the seven creatures actually need (from the API)

| Creature | Beyond what the env supports today |
|---|---|
| Bandit | Long range on the crossbow (80/320 ft) |
| Cat | Stealth +4 (climb speed is moot without terrain) |
| Commoner | Nothing — fully covered already |
| Goblin | **Nimble Escape** (Disengage *or Hide* as a bonus action), Stealth +6, long range on shortbow |
| Guard | Thrown spear (20/60), versatile damage — already approximated as two attack entries |
| Scout | **Multiattack** (two melee *or* two ranged attacks), Stealth +6, long range on longbow |
| Wolf | **Bite rider: DC 11 STR save or knocked prone**, **Pack Tactics** (needs an ally), Stealth +4 |

Standard actions every creature gets that the env lacks: **Hide**, **Search**,
**Shove**, **Grapple** (and escaping one), Help, Ready, Use an Object. Help is
meaningless without allies, Ready needs a trigger system, and there are no
objects — those three are deferred alongside spells.

## Design decisions

### 1. Conditions as first-class state

Add a `conditions: set[str]` to `Creature` (initially `prone`, `grappled`,
`hidden` — hidden is technically a state, not a condition, but it behaves like
one). All attack-roll modifiers flow through **one resolution function** that
collects every advantage source and every disadvantage source into two
booleans and applies the real 5e stacking rule (any advantage + any
disadvantage = straight roll — the current pairwise cancellation is wrong once
three or more sources exist):

- advantage: target prone (melee only), attacker hidden, Pack Tactics
- disadvantage: target prone (ranged), attacker prone, target dodging,
  ranged attack while adjacent, long range, target hidden

### 2. Ability scores and checks in the data model

Stat blocks gain `abilities` (STR–CHA scores), `skills` (e.g.
`{"stealth": 6, "perception": 5}`), and `passive_perception` — all straight
from the API. From these derive: saving throws (wolf bite), Stealth checks
(Hide), and contested checks (Shove/Grapple: Athletics vs Athletics/Acrobatics
— none of these seven have Athletics, so raw STR/DEX modifiers apply).

### 3. Attacks get riders, real ranges, and Multiattack

Attack dicts gain:

- `range_normal` / `range_long` (cells; attacks between them roll at
  disadvantage, beyond long range are invalid)
- `on_hit_effects`, e.g. the wolf bite:
  `[{"save": {"ability": "str", "dc": 11}, "on_fail": {"condition": "prone"}}]`
- Multiattack via an **attack-slot counter**: taking the Attack action grants
  `num_attacks` slots (scout: 2) that are spent one micro-action at a time —
  this fits the existing micro-action turn structure and preserves per-attack
  decisions. Enforce the scout's "two melee or two ranged" by locking the
  slot type to the first attack made that turn. (Fallback if this proves hard
  to learn: one action = the whole multiattack sequence.)

### 4. Hide without terrain (abstract line-of-sight)

The grid is featureless, so true hiding is impossible RAW. Abstract rule for
now: a creature may attempt Hide if it is **not adjacent** to its enemy; roll
Stealth vs the enemy's passive Perception. While hidden: attacks against you
have disadvantage, your first attack has advantage and reveals you, and
moving adjacent reveals you. **Search** (action) re-rolls perception actively
(Keen traits grant advantage here) to strip `hidden`. Positions stay fully
observable to the RL agent — hiding affects dice, not information. This is a
deliberate simplification; real obstacles/cover are a later phase.

### 5. Prone, Shove, Grapple

- `prone`: attacker disadvantage on its own attacks; melee vs it advantage,
  ranged vs it disadvantage; **stand_up** is a new action costing half speed
  (movement, not the Action).
- **Shove** (replaces one attack slot): contested STR check; on win, choose
  knock prone or push 1 cell (model as two discrete actions: `shove_prone`,
  `shove_push`).
- **Grapple** (replaces one attack slot): contested check; grappled target's
  speed becomes 0; `escape_grapple` is an action using the same contest.

### 6. Traits as declarative hooks

Stat blocks gain `traits: ["pack_tactics", "nimble_escape", ...]` consumed by
the resolution function and the action-map builder:

- `nimble_escape` → adds `bonus_disengage` and `bonus_hide` to the goblin's
  action map (the existing placeholder strings are replaced).
- `pack_tactics` → wired into advantage resolution but **inert in 1v1**; it
  activates when multi-creature combat lands (later phase).
- `keen_*` → advantage on Search checks.

### 7. Action map after the change (per creature)

4 moves + N attacks + `stand_up` + Dash/Disengage/Dodge/Hide/Search (5) +
`shove_prone`/`shove_push`/`grapple`/`escape_grapple` (4) + trait bonus
actions + `end_turn`. Goblin: 17 actions; scout: 16. Still comfortably small
for DQN. Two consequences worth handling:

- **All saved models become incompatible** (again). Fine — retrain.
- More actions = more invalid choices early in training. Consider exposing an
  action mask (`env.action_masks()`) and trying `MaskablePPO` from
  `sb3-contrib`; keep the −0.05 invalid-action penalty for DQN. Also consider
  dropping the +0.05 Dash/Dodge/Disengage shaping bonus — with a richer
  action space it invites reward hacking, and the real value already flows
  through damage/win rewards.

### 8. Observation additions

Per side: `is_prone`, `is_grappled`, `is_hidden`, plus `attacks_remaining_norm`
for the agent and the enemy's visible economy flags (`enemy_is_dodging` is
currently observable only implicitly). Keep everything float32 Box(0,1) like
the existing flags.

## Phases

**Phase 0 — infrastructure (no new actions). ✅ DONE.** Ability
scores/skills/passive perception in `Creature` + bestiary; saving-throw
helper on top of `dice.roll`; generalized advantage/disadvantage resolution;
the `conditions` set. Unit tests for stacking rules and saves. (Contested-check
helpers deferred to Phase 3, where they are first used.)

**Phase 1 — attack fidelity. ✅ DONE.** Attack slots (Multiattack with the
melee/ranged type lock), long range with disadvantage, on-hit save riders,
`prone` + `stand_up` + double-cost crawling. This alone makes the wolf, scout,
guard, and both bow users faithful. Observation space updated (including
forward-declared hidden/grappled flags so Phases 2–3 won't break it again);
the +0.05 Dash/Dodge/Disengage shaping bonus was removed as planned.

**Phase 2 — stealth. ✅ DONE.** Hide, `hidden` state, Search (keen senses give
advantage), Nimble Escape bonus actions, unseen-attacker modifiers, reveal on
attacking or adjacency. Goblin and cat are now faithful. The rule-based AI
searches when its target is hidden.

**Phase 3 — contests. ✅ DONE.** Shove (prone/push) and Grapple/escape as
attack-slot consumers with contested checks (ties to the defender); grapples
auto-release when the grappler leaves reach. Completes the no-ally, no-object
subset of standard actions.

**Phase 4 — SRD API bestiary generator.** `bestiary/fetch_srd.py` that pulls
any monster by index from `dnd5eapi.co`, converts to the env stat format
(structured fields cover attack bonus, damage dice, save DCs, multiattack
composition, skills, senses), and caches JSON into `bestiary/srd/`. The seven
hand-written modules become thin wrappers over generated data; a test compares
generated vs. expected blocks. This is what makes the bestiary scale.

**Phase 5 — deferred.** Multi-creature combat (activates Pack Tactics and
Help; large obs/action redesign for targeting), terrain/obstacles (real cover,
line of sight, the cat's climb speed), Ready, opportunity-attack choices,
critical hits (nat 20 = double dice — small and could be pulled into Phase 1),
and spells.

Each phase ends green: full pytest suite + a short training smoke run
(random-policy baseline vs. ~30k-step DQN win rate) to confirm the new
mechanics didn't break learnability.
