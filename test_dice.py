import pytest
from dice import roll # Assuming dice.py is in the same directory or PYTHONPATH

# Test cases for valid dice strings
VALID_ROLLS_SPECS = [
    # dice_string, min_expected, max_expected, num_dice, dice_type
    ("1d6", 1, 6, 1, 6),
    ("d6", 1, 6, 1, 6), # Default to 1 die
    ("2d4", 2, 8, 2, 4),
    ("1d8+2", 3, 10, 1, 8),
    ("3d10+5", 8, 35, 3, 10),
    ("1d20-1", 0, 19, 1, 20),
    ("1d4-4", -3, 0, 1, 4), # Result can be negative
    ("10d100", 10, 1000, 10, 100),
    ("1d12+0", 1, 12, 1, 12),
    ("1d8-0", 1, 8, 1, 8),
    ("0d6", 0, 0, 0, 6), # 0 dice should result in 0
    ("0d10+5", 5, 5, 0, 10), # 0 dice + modifier
    ("0d20-3", -3, -3, 0, 20), # 0 dice - modifier
]

@pytest.mark.parametrize("dice_string, min_expected, max_expected, num_dice_unused, dice_type_unused", VALID_ROLLS_SPECS)
def test_roll_valid_ranges(dice_string, min_expected, max_expected, num_dice_unused, dice_type_unused):
    """Test that rolls fall within the expected range."""
    # Perform multiple rolls to increase chance of catching issues
    for _ in range(100): 
        result = roll(dice_string)
        assert min_expected <= result <= max_expected

def test_roll_single_number_input():
    """Test that a string containing just a number returns that number."""
    assert roll("5") == 5
    assert roll("0") == 0
    assert roll("-3") == -3
    assert roll("100") == 100

# Test cases for invalid dice strings
INVALID_ROLL_STRINGS = [
    "abc",
    "1d", # Missing dice type
    "d",  # Missing dice type
    "1k6", # Invalid separator
    "2d6!", # Invalid character
    "1d6+a", # Invalid modifier value
    "1d6-b", # Invalid modifier value
    "1d6x2", # Invalid modifier operator
    "", # Empty string
    "+5", # Modifier only (explicit plus is treated as invalid for plain numbers)
    # "-2", # Removed: "-2" is now treated as a valid single number input (value -2)
    "d6+", # Incomplete modifier
    "1dd6", # Double 'd'
    "1d6+-2" # Double operator
]

@pytest.mark.parametrize("invalid_string", INVALID_ROLL_STRINGS)
def test_roll_invalid_input(invalid_string):
    """Test that invalid dice strings raise ValueError."""
    with pytest.raises(ValueError):
        roll(invalid_string)

def test_roll_consistency_for_zero_dice():
    """Test specific cases for zero dice."""
    assert roll("0d6") == 0
    assert roll("0d10+5") == 5
    assert roll("0d20-3") == -3

def test_roll_default_one_die():
    """Test that 'dX' format defaults to '1dX'."""
    for _ in range(20): # Check a few times for randomness
        assert 1 <= roll("d6") <= 6
        assert 1 <= roll("d20") <= 20
        assert 1 <= roll("d10+2") <= 12
        assert 0 <= roll("d4-1") <= 3

# More detailed check for a specific complex roll
def test_roll_specific_complex_case():
    """Test a specific complex roll like 3d6+5 multiple times."""
    # 3d6+5: min = (1*3)+5 = 8, max = (6*3)+5 = 23
    min_expected = 8
    max_expected = 23
    for _ in range(100):
        result = roll("3d6+5")
        assert min_expected <= result <= max_expected

# Test for large numbers of dice and large modifiers
def test_roll_large_values():
    """Test rolls with large numbers of dice, types, and modifiers."""
    # 100d100+100: min = (1*100)+100 = 200, max = (100*100)+100 = 10100
    min_expected = 200
    max_expected = 10100
    # Run fewer iterations as it's a larger roll, but still check a few times
    for _ in range(10):
        result = roll("100d100+100")
        assert min_expected <= result <= max_expected

    # 50d20-50: min = (1*50)-50 = 0, max = (20*50)-50 = 950
    min_expected = 0
    max_expected = 950
    for _ in range(10):
        result = roll("50d20-50")
        assert min_expected <= result <= max_expected

# Ensure the function exists and is callable
def test_roll_function_exists():
    assert callable(roll)

if __name__ == "__main__":
    # This block allows running pytest directly on this file if needed
    # For example, if you want to run it without the pytest command
    # pytest.main() 
    # However, it's more standard to run `pytest` from the terminal
    print("To run these tests, navigate to the directory containing")
    print("dice.py and test_dice.py, then run the command: pytest")

# Example of how one might test the distribution if truly needed,
# but this is more complex and usually overkill for unit tests.
# For now, range checking is sufficient.
# def test_roll_distribution_roughly(dice_string="1d6"):
#     counts = {}
#     num_rolls = 10000
#     for _ in range(num_rolls):
#         res = roll(dice_string)
#         counts[res] = counts.get(res, 0) + 1
#     # Basic check: ensure all possible outcomes appeared for a simple die
#     if dice_string == "1d6":
#         for i in range(1, 7):
#             assert i in counts
#     # Further statistical tests could be applied here (e.g., chi-squared)
#     # but that's generally beyond typical unit testing.
#     print(f"Distribution for {dice_string} over {num_rolls} rolls: {sorted(counts.items())}")
