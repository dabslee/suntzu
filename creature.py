import dice
import verbage

class Action():
    is_attack = False
    is_spell = False

class Creature():
    # Trait variables
    traits = {
        "creature_type" : verbage.CreatureTypes.HUMANOIDS,
        "ac" : 10,
        "hp" : 4,
        "hit_dice_count" : 1,
        "hit_dice_size" : 8,
        "speed" : 30,
        "ability_scores" : [10,10,10,10,10,10,],
        "saving_throw_bonuses" : [0,0,0,0,0,0,],
        "skill_bonuses" : {},
        "vulnerabilities" : [],
        "resistances" : [],
        "immunities" : [],
        "senses" : [("sight", 5280)],
        "cr" : 0
    }

    # State variables
    state = {
        "position" : [0,0,0],
        "conditions" : []
    }

    # Basic methods
    def ability_score_mod(self, ability_score):
        index = verbage.AbilityScores.index_of_score(ability_score)
        return (self.ability_scores[index]-10)/2

    def roll_save(self, ability_score):
        index = verbage.AbilityScores.index(ability_score)
        return dice.roll(f"1d20t+{self.ability_score_mod(ability_score)}+{self.saving_throw_bonus[index]}")
    
    def roll_ability_check(self, ability_score):
        index = verbage.AbilityScores.index(ability_score)
        return dice.roll(f'1d20t+{(self.ability_scores[index]-10)/2}')
    
    def roll_skill_check(self, skill):
        return dice.roll(f'1d20t+{self.skill_mods[skill]}')
    
    # Actions
    actions = {}
    bonus_actions = {}
    reactions = {}
    legendary_actions = {}
    lair_actions = {}
    free = {}
    movement = {}