# bestiary/scout.py — SRD Scout (CR 1/2)
import os

def get_scout_stats():
    return {
        "display_name": "Scout",
        "token_image": os.path.join(os.path.dirname(__file__), "tokens", "Scout.webp"),
        "ac": 13,
        "max_hp": 16,  # Avg of 3d8+3
        "speed_total": 6,  # 30ft / 5ft per cell
        "abilities": {"str": 11, "dex": 14, "con": 12, "int": 11, "wis": 13, "cha": 11},
        "skills": {"nature": 4, "perception": 5, "stealth": 6, "survival": 5},
        "passive_perception": 15,
        "traits": ["keen_hearing_and_sight"],
        "attacks": [
            {
                "name": "shortsword",
                "to_hit": 4,
                "damage_dice": "1d6+2",
                "range": 1, # 5ft
                "attack_type": "melee",
                "num_attacks": 2  # Multiattack: two melee attacks
            },
            {
                "name": "longbow",
                "to_hit": 4,
                "damage_dice": "1d8+2",
                "range": 30,        # 150ft normal range
                "range_long": 120,  # 600ft long range (disadvantage beyond normal)
                "attack_type": "ranged",
                "num_attacks": 2  # Multiattack: two ranged attacks
            }
        ],
        "bonus_actions": []
    }
