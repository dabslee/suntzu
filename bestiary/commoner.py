# bestiary/commoner.py
def get_commoner_stats():
    return {
        "ac": 10,
        "max_hp": 4,
        "speed_total": 6,  # 30ft / 5ft per cell
        "attacks": [
            {"name": "club", "to_hit": 2, "damage_dice": "1d4", "num_attacks": 1, "range": 1} # Added range
        ],
        "bonus_actions": []
    }
