# bestiary/bandit.py
def get_bandit_stats():
    return {
        "ac": 12,
        "max_hp": 11,  # Avg of 2d8+2
        "speed_total": 6,  # 30ft / 5ft per cell
        "attacks": [
            {
                "name": "scimitar",
                "to_hit": 3,
                "damage_dice": "1d6+1",
                "range": 1, # 5ft
                "attack_type": "melee",
                "num_attacks": 1
            },
            {
                "name": "light_crossbow",
                "to_hit": 3,
                "damage_dice": "1d8+1",
                "range": 16, # 80ft
                "attack_type": "ranged",
                "num_attacks": 1
            }
        ],
        "bonus_actions": []
    }
