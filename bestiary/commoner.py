# bestiary/commoner.py — SRD Commoner (CR 0)
import os

def get_commoner_stats():
    return {
        "display_name": "Commoner",
        "token_image": os.path.join(os.path.dirname(__file__), "tokens", "Commoner.webp"),
        "ac": 10,
        "max_hp": 4,
        "speed_total": 6,  # 30ft / 5ft per cell
        "abilities": {"str": 10, "dex": 10, "con": 10, "int": 10, "wis": 10, "cha": 10},
        "skills": {},
        "passive_perception": 10,
        "traits": [],
        "attacks": [
            {"name": "club", "to_hit": 2, "damage_dice": "1d4", "num_attacks": 1, "range": 1, "attack_type": "melee"}
        ],
        "bonus_actions": []
    }
