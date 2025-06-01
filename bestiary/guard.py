# bestiary/guard.py
def get_guard_stats():
    return {
        "ac": 16,
        "max_hp": 11,  # Avg of 2d8+2
        "speed_total": 6,  # 30ft / 5ft per cell
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
                "name": "spear_ranged",
                "to_hit": 3,
                "damage_dice": "1d6+1",
                "range": 4, # 20ft
                "attack_type": "ranged",
                "num_attacks": 1
            }
        ],
        "bonus_actions": []
    }
