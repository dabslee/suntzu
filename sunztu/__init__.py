"""sunztu — RL agents for D&D 5e-style grid combat.

Package layout (Phase A structure):
- sunztu.dice      — dice-string roller ("2d20kh1" etc.)
- sunztu.creature  — the Creature data model (stats, conditions, saves)
- sunztu.models    — algorithm-agnostic saved-policy loading (MaskablePPO/DQN)
- sunztu.env       — DnDCombatEnv, the Gymnasium environment
- sunztu.render    — pygame rendering (only imported when a window is needed)
- sunztu.spectator — symmetric two-agent driving loop (watching + head-to-head eval)
- sunztu.training  — shared vec-env/hyperparameter helpers for the trainers

Creature stat blocks live in the top-level `bestiary` package.
"""
from sunztu.dice import roll
from sunztu.creature import Creature
from sunztu.models import load_model_any, MASKABLE_AVAILABLE
from sunztu.env import DnDCombatEnv

__all__ = ["roll", "Creature", "load_model_any", "MASKABLE_AVAILABLE", "DnDCombatEnv"]
