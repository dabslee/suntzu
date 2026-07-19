# bestiary/cat.py — SRD Cat (CR 0)
import os

def get_cat_stats():
    return {
        "display_name": "Cat",
        "token_image": os.path.join(os.path.dirname(__file__), "tokens", "Cat.webp"),
        "ac": 12,
        "max_hp": 2,  # Avg of 1d4
        "speed_total": 8,  # 40ft / 5ft per cell (climb 30ft not modeled — no terrain)
        "abilities": {"str": 3, "dex": 15, "con": 10, "int": 3, "wis": 12, "cha": 7},
        "skills": {"perception": 3, "stealth": 4},
        "passive_perception": 13,
        "traits": ["keen_smell"],
        "attacks": [
            {
                "name": "claws",
                "to_hit": 2,  # Assumed DEX-based: +0 proficiency, +2 DEX
                "damage_dice": "1", # Flat 1 damage
                "range": 1, # 5ft
                "attack_type": "melee",
                "num_attacks": 1
            }
        ],
        "bonus_actions": []
    }
