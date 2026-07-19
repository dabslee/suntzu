"""Algorithm-agnostic loading of saved policies.

Training currently produces MaskablePPO checkpoints; older runs produced DQN.
Both are plain SB3 zips (policy weights only — no env code is pickled), so
either kind can serve as an opponent or an evaluation subject.
"""
from typing import Any

try:
    from stable_baselines3 import DQN
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    class DQN:  # placeholder so isinstance checks stay valid without SB3
        pass
    print("Warning: stable-baselines3 not found. DRL agent functionality will be limited.")

# sb3-contrib (MaskablePPO) is optional; DQN checkpoints keep working without it.
try:
    from sb3_contrib import MaskablePPO
    MASKABLE_AVAILABLE = True
except ImportError:
    MASKABLE_AVAILABLE = False
    class MaskablePPO:  # placeholder so isinstance checks stay valid without sb3-contrib
        pass


def load_model_any(path: str, device: str = 'auto') -> Any:
    """Load a saved policy zip regardless of algorithm (MaskablePPO or DQN)."""
    errors = []
    if MASKABLE_AVAILABLE:
        try:
            return MaskablePPO.load(path, device=device)
        except Exception as e:
            errors.append(f"MaskablePPO: {e}")
    if SB3_AVAILABLE:
        try:
            return DQN.load(path, device=device)
        except Exception as e:
            errors.append(f"DQN: {e}")
    raise RuntimeError(f"Could not load model from {path}: {'; '.join(errors) or 'SB3 not installed'}")


def uses_action_masks(model: Any) -> bool:
    """True when a loaded model expects action_masks= in predict()."""
    return MASKABLE_AVAILABLE and isinstance(model, MaskablePPO)
