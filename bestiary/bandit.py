# bestiary/bandit.py — SRD Bandit (CR 1/8)
import os

def get_bandit_stats():
    return {
        "display_name": "Bandit",
        "token_image": os.path.join(os.path.dirname(__file__), "tokens", "Bandit.webp"),
        "ac": 12,
        "max_hp": 11,  # Avg of 2d8+2
        "speed_total": 6,  # 30ft / 5ft per cell
        "abilities": {"str": 11, "dex": 12, "con": 12, "int": 10, "wis": 10, "cha": 10},
        "skills": {},
        "passive_perception": 10,
        "traits": [],
        "attacks": [
            {
                "name": "scimitar",
                "to_hit": 3,
                "damage_dice": "1d6+1",
                "range": 1,  # 5ft
                "attack_type": "melee",
                "num_attacks": 1
            },
            {
                "name": "light_crossbow",
                "to_hit": 3,
                "damage_dice": "1d8+1",
                "range": 16,       # 80ft normal range
                "range_long": 64,  # 320ft long range (disadvantage beyond normal)
                "attack_type": "ranged",
                "num_attacks": 1
            }
        ],
        "bonus_actions": []
    }
