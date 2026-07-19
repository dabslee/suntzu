# bestiary/guard.py — SRD Guard (CR 1/8)
import os

def get_guard_stats():
    return {
        "display_name": "Guard",
        "token_image": os.path.join(os.path.dirname(__file__), "tokens", "Guard.webp"),
        "ac": 16,
        "max_hp": 11,  # Avg of 2d8+2
        "speed_total": 6,  # 30ft / 5ft per cell
        "abilities": {"str": 13, "dex": 12, "con": 12, "int": 10, "wis": 11, "cha": 10},
        "skills": {"perception": 2},
        "passive_perception": 12,
        "traits": [],
        "attacks": [
            {
                "name": "spear_melee",
                "to_hit": 3,
                "damage_dice": "1d6+1", # One-handed with shield
                "range": 1, # 5ft
                "attack_type": "melee",
                "num_attacks": 1
            },
            {
                "name": "spear_thrown",
                "to_hit": 3,
                "damage_dice": "1d6+1",
                "range": 4,        # 20ft normal range
                "range_long": 12,  # 60ft long range (disadvantage beyond normal)
                "attack_type": "ranged",
                "num_attacks": 1
            }
        ],
        "bonus_actions": []
    }
