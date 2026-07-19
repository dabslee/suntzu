import random
import re

def roll(dice_string: str) -> int:
    """
    Rolls dice based on a D&D-style dice string.

    Args:
        dice_string: The dice string (e.g., "1d8", "2d6+2", "1d20-1").

    Returns:
        The total result of the dice roll.
    """
    # New pattern: r"(\d*)d(\d+)(?:(kh|kl)(\d+))?(?:([+-])(\d+))?"
    # Groups:
    # 1: num_dice_str
    # 2: dice_type_str
    # 3: keep_instr (kh or kl)
    # 4: keep_count_str
    # 5: modifier_sign (+ or -)
    # 6: modifier_val_str
    pattern = re.compile(r"(\d*)d(\d+)(?:(kh|kl)(\d+))?(?:([+-])(\d+))?")
    match = pattern.fullmatch(dice_string)

    if not match:
        if re.fullmatch(r"-?\d+", dice_string):
            return int(dice_string)
        else:
            raise ValueError(f"Invalid dice string: {dice_string}")

    num_dice_str, dice_type_str, keep_instr, keep_count_str, modifier_sign, modifier_val_str = match.groups()

    num_dice = int(num_dice_str) if num_dice_str else 1
    dice_type = int(dice_type_str)

    if dice_type <= 0:
        raise ValueError("Dice type must be positive.")
    if num_dice < 0:
        raise ValueError("Number of dice cannot be negative.")

    total_roll = 0

    if keep_instr and keep_count_str:
        keep_count = int(keep_count_str)
        if keep_count <= 0:
            raise ValueError("Keep count must be positive.")
        if keep_count > num_dice:
            raise ValueError(f"Cannot keep {keep_count} dice from {num_dice} dice.")

        rolls = []
        for _ in range(num_dice):
            rolls.append(random.randint(1, dice_type))

        if keep_instr == "kh":
            rolls.sort(reverse=True)
        elif keep_instr == "kl":
            rolls.sort()

        total_roll = sum(rolls[:keep_count])
    else:
        # Standard roll
        if num_dice == 0:
            total_roll = 0 # 0d6 = 0
        else:
            for _ in range(num_dice):
                total_roll += random.randint(1, dice_type)

    if modifier_sign and modifier_val_str:
        modifier_val = int(modifier_val_str)
        if modifier_sign == '+':
            total_roll += modifier_val
        else:  # modifier_sign == '-'
            total_roll -= modifier_val

    return total_roll

if __name__ == '__main__':
    # Example usage:
    print(f"Rolling 1d8+3: {roll('1d8+3')}")
    print(f"Rolling 2d6: {roll('2d6')}")
    print(f"Rolling 1d4-1: {roll('1d4-1')}")
    print(f"Rolling 1d20: {roll('1d20')}")
    print(f"Rolling 3d6+5: {roll('3d6+5')}")
    print(f"Rolling 10d4-10: {roll('10d4-10')}")
    # Example of just a number
    print(f"Rolling 5: {roll('5')}")
    try:
        roll("invalid")
    except ValueError as e:
        print(e)
    try:
        roll("d6") # Handled by making num_dice default to 1
    except ValueError as e:
        print(e)
    print(f"Rolling d6: {roll('d6')}")

    # Test edge cases for modifiers
    print(f"Rolling 1d20+0: {roll('1d20+0')}")
    print(f"Rolling 1d20-0: {roll('1d20-0')}")
    
    # Test different dice types
    dice_types_to_test = ["1d4", "1d6", "1d8", "1d10", "1d12", "1d20", "1d100"]
    for dt_string in dice_types_to_test:
        print(f"Rolling {dt_string}: {roll(dt_string)}")

    # Test multiple dice
    print(f"Rolling 0d6: {roll('0d6')}") # Should be 0 + modifier if any
    print(f"Rolling 0d6+5: {roll('0d6+5')}")

    # Test empty string (should raise error)
    try:
        roll("")
    except ValueError as e:
        print(f"Rolling '': {e}")
    
    # Test string with only modifier (should raise error as per current regex, or be handled if we want to support it)
    # Current regex does not support "+5" or "-2" alone
    try:
        roll("+5")
    except ValueError as e:
        print(f"Rolling '+5': {e}")
    try:
        roll("-2")
    except ValueError as e:
        print(f"Rolling '-2': {e}")

    # Test with large numbers
    print(f"Rolling 100d100+100: {roll('100d100+100')}")
