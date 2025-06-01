# bestiary/cat.py
def get_cat_stats():
    return {
        "ac": 12,
        "max_hp": 2,  # Avg of 1d4
        "speed_total": 8,  # 40ft / 5ft per cell
        "attacks": [
            {
                "name": "claws",
                "to_hit": 2,  # Assumed DEX-based: +0 proficiency, +2 DEX
                "damage_dice": "1", # Flat 1 damage
                "range": 1, # 5ft
                "attack_type": "melee",
                "num_attacks": 1
            }
        ],
        "bonus_actions": []
    }
