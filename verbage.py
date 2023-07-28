from enum import Enum

class AbilityScores(Enum):
    STR = "STRENGTH"
    DEX = "DEXTERITY"
    CON = "CONSTITUTION"
    INT = "INTELLIGENCE"
    WIS = "WISDOM"
    CHA = "CHARISMA"
    
    listed = [STR, DEX, CON, INT, WIS, CHA]
    def score_at_index(num):
        return AbilityScores.listed[num]
    def index_of_score(score):
        return AbilityScores.listed.index(score)

class DamageTypes(Enum):
    ACID = "ACID"
    BLUDGEONING = "BLUDGEONING"
    COLD = "COLD"
    FIRE = "FIRE"
    FORCE = "FORCE"
    LIGHTNING = "LIGHTNING"
    NECROTIC = "NECROTIC"
    PIERCING = "PIERCING"
    POISON = "POISON"
    PSYCHIC = "PSYCHIC"
    RADIANT = "RADIANT"
    SLASHING = "SLASHING"
    THUNDER = "THUNDER"

class CreatureTypes(Enum):
    ABERRATIONS = "ABERRATIONS"
    BEASTS = "BEASTS"
    CELESTIALS = "CELESTIALS"
    CONSTRUCTS = "CONSTRUCTS"
    DRAGONS = "DRAGONS"
    ELEMENTALS = "ELEMENTALS"
    FEY = "FEY"
    FIENDS = "FIENDS"
    GIANTS = "GIANTS"
    HUMANOIDS = "HUMANOIDS"
    MONSTROSITIES = "MONSTROSITIES"
    OOZES = "OOZES"
    PLANTS = "PLANTS"
    UNDEAD = "UNDEAD"