# bestiary/goblin.py
def get_goblin_stats():
    return {
        "ac": 15,
        "max_hp": 7,  # Avg of 2d6
        "speed_total": 6,  # 30ft / 5ft per cell
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
                "range": 16, # 80ft
                "attack_type": "ranged",
                "num_attacks": 1
            }
        ],
        # Nimble Escape: Disengage or Hide as bonus action.
        # Hide is not implemented as a game mechanic yet.
        # bonus_action_disengage would need specific handling in step()
        "bonus_actions": ["nimble_escape_disengage", "nimble_escape_hide"]
    }
