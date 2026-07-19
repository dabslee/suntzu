# bestiary/wolf.py — SRD Wolf (CR 1/4)
import os

def get_wolf_stats():
    return {
        "display_name": "Wolf",
        "token_image": os.path.join(os.path.dirname(__file__), "tokens", "Wolf.webp"),
        "ac": 13,
        "max_hp": 11,
        "speed_total": 8,  # 40ft / 5ft per cell
        "abilities": {"str": 12, "dex": 15, "con": 12, "int": 3, "wis": 12, "cha": 6},
        "skills": {"perception": 3, "stealth": 4},
        "passive_perception": 13,
        # pack_tactics is declared but inert until multi-creature combat exists.
        "traits": ["keen_hearing_and_smell", "pack_tactics"],
        "attacks": [
            {
                "name": "bite",
                "to_hit": 4,
                "damage_dice": "2d4+2",
                "range": 1,
                "attack_type": "melee",
                "num_attacks": 1,
                # On hit: DC 11 STR save or the target is knocked prone.
                "on_hit_effects": [
                    {"save": {"ability": "str", "dc": 11},
                     "on_fail": {"condition": "prone"}}
                ]
            }
        ],
        "bonus_actions": []
    }
