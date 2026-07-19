"""Backwards-compatibility shim. The environment moved to sunztu.env.

Import from the package instead:

    from sunztu.env import DnDCombatEnv
    from sunztu.creature import Creature
    from sunztu.models import load_model_any
"""
import warnings

from sunztu.creature import Creature, DEFAULT_ABILITIES
from sunztu.dice import roll
from sunztu.env import DnDCombatEnv
from sunztu.models import DQN, MASKABLE_AVAILABLE, MaskablePPO, SB3_AVAILABLE, load_model_any

warnings.warn(
    "dnd_gym_env is deprecated; import from sunztu.env / sunztu.creature / "
    "sunztu.models instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DnDCombatEnv", "Creature", "DEFAULT_ABILITIES", "roll",
           "load_model_any", "MASKABLE_AVAILABLE", "MaskablePPO", "DQN",
           "SB3_AVAILABLE"]
