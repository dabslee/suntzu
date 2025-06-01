# bestiary/scout.py
def get_scout_stats():
    return {
        "ac": 13,
        "max_hp": 16,  # Avg of 3d8+3
        "speed_total": 6,  # 30ft / 5ft per cell
        "attacks": [
            {
                "name": "shortsword",
                "to_hit": 4,
                "damage_dice": "1d6+2",
                "range": 1, # 5ft
                "attack_type": "melee",
                "num_attacks": 1 # Multiattack not fully supported by env step logic yet
            },
            {
                "name": "longbow",
                "to_hit": 4,
                "damage_dice": "1d8+2",
                "range": 30, # 150ft
                "attack_type": "ranged",
                "num_attacks": 1 # Multiattack not fully supported by env step logic yet
            }
        ],
        "bonus_actions": []
    }
