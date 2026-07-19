# bestiary/__init__.py
import importlib

from .commoner import get_commoner_stats
from .wolf import get_wolf_stats
from .cat import get_cat_stats
from .bandit import get_bandit_stats
from .guard import get_guard_stats
from .goblin import get_goblin_stats
from .scout import get_scout_stats

CREATURES = ["bandit", "cat", "commoner", "goblin", "guard", "scout", "wolf"]


def get_stats(creature_name: str) -> dict:
    """Stat block by creature name, e.g. get_stats('wolf')."""
    return globals()[f"get_{creature_name}_stats"]()


def load_stats_from_module(module_path_str: str) -> dict:
    """Import a stats module by dotted path (e.g. 'bestiary.wolf') and call
    its get_<name>_stats(). Used by CLIs that take stats modules as args."""
    creature = module_path_str.split('.')[-1]
    expected_func_name = f"get_{creature}_stats"
    try:
        module = importlib.import_module(module_path_str)
        stats_func = getattr(module, expected_func_name)
    except ImportError:
        print(f"Error: could not import module '{module_path_str}'.")
        raise
    except AttributeError:
        print(f"Error: could not find '{expected_func_name}' in '{module_path_str}'.")
        raise
    print(f"Loaded stats function '{expected_func_name}' from '{module_path_str}'")
    return stats_func()


__all__ = [
    "get_commoner_stats",
    "get_wolf_stats",
    "get_cat_stats",
    "get_bandit_stats",
    "get_guard_stats",
    "get_goblin_stats",
    "get_scout_stats",
    "get_stats",
    "load_stats_from_module",
    "CREATURES",
]
