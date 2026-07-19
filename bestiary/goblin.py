# bestiary/goblin.py — SRD Goblin (CR 1/4)
import os

def get_goblin_stats():
    return {
        "display_name": "Goblin",
        "token_image": os.path.join(os.path.dirname(__file__), "tokens", "Goblin.webp"),
        "ac": 15,
        "max_hp": 7,  # Avg of 2d6
        "speed_total": 6,  # 30ft / 5ft per cell
        "abilities": {"str": 8, "dex": 14, "con": 10, "int": 10, "wis": 8, "cha": 8},
        "skills": {"stealth": 6},
        "passive_perception": 9,
        # Nimble Escape: Disengage or Hide as a bonus action. Declared as a
        # trait; the bonus-action wiring lands with the stealth phase
        # (the old placeholder bonus_actions strings had no effect).
        "traits": ["nimble_escape"],
        "attacks": [
            {
                "name": "scimitar",
                "to_hit": 4,
                "damage_dice": "1d6+2",
                "range": 1, # 5ft
                "attack_type": "melee",
                "num_attacks": 1
            },
            {
                "name": "shortbow",
                "to_hit": 4,
                "damage_dice": "1d6+2",
                "range": 16,       # 80ft normal range
                "range_long": 64,  # 320ft long range (disadvantage beyond normal)
                "attack_type": "ranged",
                "num_attacks": 1
            }
        ],
        "bonus_actions": []
    }
