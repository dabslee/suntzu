import sys
from dnd_gym_env import DnDCombatEnv # Assuming dnd_gym_env.py is in the same directory or PYTHONPATH
from gymnasium.utils.env_checker import check_env

def run():
    """
    Runs the D&D Combat Environment with basic checks and interaction.
    """
    print("Initializing D&D Combat Environment...")

    agent_stats = {
        "ac": 15,
        "max_hp": 50,
        "speed_total": 6, # cells
        "attacks": [
            {"name": "longsword", "to_hit": 5, "damage_dice": "1d8+3", "num_attacks": 1},
            {"name": "shortbow", "to_hit": 4, "damage_dice": "1d6+2", "num_attacks": 1}
        ],
        "bonus_actions": ["bonus_move_1_cell"]
        # initial_position is handled by the environment's reset method
    }

    enemy_stats = {
        "ac": 13,
        "max_hp": 30,
        "speed_total": 5, # cells
        "attacks": [
            {"name": "scimitar", "to_hit": 4, "damage_dice": "1d6+2", "num_attacks": 1}
        ],
        "bonus_actions": []
        # initial_position is handled by the environment's reset method
    }

    try:
        env = DnDCombatEnv(map_width=10, map_height=10, 
                           agent_stats=agent_stats, 
                           enemy_stats=enemy_stats)
    except Exception as e:
        print(f"Error during environment instantiation: {e}")
        print("Please ensure dice.py is accessible and correct.")
        return

    print("Environment instantiated.")

    # Run Environment Checker
    try:
        print("\nRunning Environment Checker...")
        # Use env.unwrapped to get the core environment if it's wrapped (e.g. by TimeLimit)
        # For a custom env not yet wrapped, env itself is fine.
        # However, check_env often expects the unwrapped version.
        check_env(env.unwrapped) # No-op if not wrapped, else gets the base env.
        print("Environment check passed successfully!")
    except Exception as e:
        print(f"Environment check failed: {e}")
        print("There might be issues with the environment's adherence to the Gymnasium API.")
        # It might be useful to proceed with the interaction loop anyway for debugging,
        # or exit if strict compliance is required first.
        # For now, let's proceed.

    # Basic Interaction Loop
    print("\nStarting basic interaction loop...")
    try:
        obs, info = env.reset(seed=42) # Using a seed for reproducibility
    except Exception as e:
        print(f"Error during env.reset(): {e}")
        return
        
    print("Initial state rendered:")
    env.render()
    print(f"Initial Observation: {obs}")
    print(f"Initial Info: {info}")

    max_steps = 30
    for step_num in range(max_steps):
        print(f"\n--- Step {step_num + 1} ---")

        if not env.agent.is_alive and not env.enemy.is_alive:
            print("Both agent and enemy are not alive. This might be an unexpected state.")
            break
        if not env.agent.is_alive:
            print("Agent is no longer alive.")
            # Loop might continue to see enemy actions or until termination condition is met
            # For now this is fine, termination should handle it.

        action = env.action_space.sample()
        
        # Get action description for printing
        try:
            action_desc = env.action_map[action] # Accessing through property
            print(f"Chosen Action (Index {action}): {action_desc.get('name', 'Unknown Action Name')} ({action_desc['type']})")
        except IndexError:
            print(f"Chosen Action (Index {action}): Invalid action index!")
            action_desc = {"type": "invalid", "name": "invalid"}


        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            print(f"Error during env.step() with action {action} ({action_desc}): {e}")
            # Depending on the error, you might want to break or try to recover.
            # For a simple run script, printing and breaking is often best.
            import traceback
            traceback.print_exc()
            break
            
        print("\nEnvironment rendered after action:")
        env.render()
        
        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}") # Should usually be False for this env
        print(f"Info: {info}")

        if terminated or truncated:
            print("\nGame Over!")
            if terminated:
                if not env.agent.is_alive: print("Agent was defeated.")
                elif not env.enemy.is_alive: print("Enemy was defeated. Agent wins!")
                else: print("Terminated for other reasons.")
            if truncated:
                print("Episode truncated (e.g., time limit reached).")
            break
    
    if step_num == max_steps - 1 and not (terminated or truncated):
        print(f"\nReached max steps ({max_steps}) without termination.")

    # Close the environment
    try:
        env.close()
        print("\nEnvironment closed.")
    except Exception as e:
        print(f"Error during env.close(): {e}")

if __name__ == '__main__':
    # Add the directory containing dnd_gym_env.py and dice.py to sys.path
    # if they are not already discoverable via PYTHONPATH
    # For example, if run_env.py is in the same directory as them:
    # import os
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # sys.path.insert(0, script_dir)
    
    # This simple structure assumes that Python's import mechanism can find
    # dnd_gym_env (and by extension, dice). This is true if:
    # 1. They are in the same directory as run_env.py and run_env.py is run from that directory.
    # 2. The directory containing them is in PYTHONPATH.
    # 3. They are installed as part of a package.
    run()
