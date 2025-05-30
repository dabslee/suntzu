# bestiary/wolf.py
def get_wolf_stats():
    return {
        "ac": 13,
        "max_hp": 11,
        "speed_total": 8,  # 40ft / 5ft per cell
        "attacks": [
            {"name": "bite", "to_hit": 4, "damage_dice": "2d4+2", "num_attacks": 1}
        ],
        "bonus_actions": [] # Wolves have Pack Tactics, but that's a more complex ability.
    }
