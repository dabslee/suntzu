# bestiary/__init__.py
from .commoner import get_commoner_stats
from .wolf import get_wolf_stats
from .cat import get_cat_stats
from .bandit import get_bandit_stats
from .guard import get_guard_stats
from .goblin import get_goblin_stats
from .scout import get_scout_stats

__all__ = [
    "get_commoner_stats",
    "get_wolf_stats",
    "get_cat_stats",
    "get_bandit_stats",
    "get_guard_stats",
    "get_goblin_stats",
    "get_scout_stats",
]
