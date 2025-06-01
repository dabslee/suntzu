import sys
import os
import argparse
from typing import Optional # For type hinting
from stable_baselines3 import DQN
from gymnasium.wrappers import FlattenObservation
from gymnasium.utils.env_checker import check_env

from dnd_gym_env import DnDCombatEnv 
from bestiary.commoner import get_commoner_stats
from bestiary.wolf import get_wolf_stats

MODEL_PATH = "dnd_dqn_agent.zip" # Default path for the agent's model

def main(agent_control_arg: str, 
         agent_model_path_arg: Optional[str], 
         enemy_model_path_arg: Optional[str], 
         export_frames_path_arg: Optional[str]):
    """
    Runs the D&D Combat Environment with specified agent and enemy control types.
    """
    print(f"\n--- Running D&D Combat Environment ---")
    print(f"Agent control: {agent_control_arg}")
    if agent_control_arg == "model":
        # Use agent_model_path_arg if provided, otherwise default to MODEL_PATH
        # This logic is slightly changed: agent_model_path_arg already has a default from argparse
        print(f"Agent model path: {agent_model_path_arg}")
    
    if enemy_model_path_arg:
        print(f"Enemy model path: {enemy_model_path_arg} (environment will attempt to load)")
    else:
        print("Enemy using rule-based AI.")
    print(f"Frame export path: {export_frames_path_arg if export_frames_path_arg else 'Disabled'}")
    print("------------------------------------")

    agent_stats = get_commoner_stats()
    enemy_stats = get_wolf_stats()

    # Create the raw environment - this instance will be used throughout.
    env = DnDCombatEnv(
        map_width=10, map_height=10,
        agent_stats=agent_stats, enemy_stats=enemy_stats,
        render_mode='human',
        export_frames_path=export_frames_path_arg,
        enemy_model_path=enemy_model_path_arg # Pass enemy model path
    )
    print("Environment instantiated.")

    # Run Environment Checker on the raw environment
    try:
        print("\nRunning Environment Checker...")
        check_env(env) 
        print("Environment check passed successfully!")
    except Exception as e:
        print(f"Environment check failed: {e}")
        env.close()
        return

    agent_model = None # For the main agent
    flatten_wrapper = FlattenObservation(env) # Wrapper for the main env, used if agent is model-controlled

    if agent_control_arg == "model":
        # Use agent_model_path_arg which has a default from argparse
        if agent_model_path_arg and os.path.exists(agent_model_path_arg):
            print(f"Loading agent model from: {agent_model_path_arg}")
            try:
                agent_model = DQN.load(agent_model_path_arg)
                print("Agent model loaded.")
            except Exception as e:
                print(f"Error loading agent model: {e}. Agent will use random actions.")
                agent_model = None # Ensure agent_model is None if loading failed
        else:
            print(f"Agent model path '{agent_model_path_arg}' not found or not specified. Agent will use random actions.")
            # agent_control_arg will remain "model" but agent_model will be None, leading to random actions.

    effective_agent_control = "model" if agent_model else "random"
    print(f"\nStarting interaction loop with {effective_agent_control} agent...")
    
    try:
        obs_dict, info = env.reset(seed=42) 
    except Exception as e:
        print(f"Error during env.reset(): {e}")
        env.close()
        return
        
    env.render() 

    max_run_steps = 150 
    for step_num in range(max_run_steps):
        print(f"\n--- Step {step_num + 1} ---")

        action = None
        action_source_info = "" # For printing the source of the action

        if agent_model: # Check if agent_model was successfully loaded
            try:
                flat_obs_for_model = flatten_wrapper.observation(obs_dict)
                action, _ = agent_model.predict(flat_obs_for_model, deterministic=True)
                action_source_info = f"Agent Model Action (Index {action}): {env._decode_action(action, is_enemy=False).get('name', 'Unknown')}"
            except Exception as e:
                print(f"Error during agent model prediction: {e}. Agent falling back to random action for this step.")
                action = env.action_space.sample()
                action_source_info = f"Agent (fallback to random) Action (Index {action}): {env._decode_action(action, is_enemy=False).get('name', 'Unknown')}"
        else: # Random agent behavior (either selected or as fallback)
            action = env.action_space.sample()
            action_source_info = f"Random Agent Action (Index {action}): {env._decode_action(action, is_enemy=False).get('name', 'Unknown')}"
        
        print(action_source_info)

        if action is None: 
            print("Error: No action determined for agent. Breaking loop.")
            break

        try:
            obs_dict, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            print(f"Error during env.step() with action {action}: {e}")
            import traceback
            traceback.print_exc()
            break
            
        env.render()
        
        print(f"Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
        print(f"Agent HP: {env.agent.current_hp}, Enemy HP: {env.enemy.current_hp}")
        print(f"Agent Pos: {env.agent.position}, Enemy Pos: {env.enemy.position}")
        print(f"Steps taken in episode: {env.current_episode_steps}")


        if terminated or truncated:
            print("\nEpisode finished!")
            if terminated:
                if env.agent.is_alive and not env.enemy.is_alive: print("Result: Agent Won!")
                elif not env.agent.is_alive: print("Result: Agent Lost!")
                else: print("Result: Terminated for other reasons.")
            if truncated:
                print(f"Result: Draw (reached max {env.max_episode_steps} steps).")
            
            # Option to run more episodes or break
            # For now, break after one episode in this script's run
            break 
    
    if step_num == max_run_steps - 1 and not (terminated or truncated):
        print(f"\nReached max run steps ({max_run_steps}) for this interaction without termination/truncation.")

    env.close()
    print("\nEnvironment closed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run D&D Combat Environment")
    parser.add_argument("--agent_control", type=str, default="random", choices=["random", "model"],
                        help="Control type for the agent. Default: random")
    parser.add_argument("--agent_model_path", type=str, default="dnd_dqn_agent.zip",
                        help="Path to agent's DRL model (used if --agent_control=model).")
    parser.add_argument("--enemy_model_path", type=str, default=None,
                        help="Path to enemy's DRL model. If None, rule-based AI is used by the environment.")
    parser.add_argument("--export_frames_path", type=str, default=None, 
                        help="Path to directory to save rendered frames. If not set, frames are not saved.")
    args = parser.parse_args()
    
    # Assuming dnd_gym_env.py, bestiary/*, and dice.py are structured to be found.
    # The call to main will be updated in the next step to use these new args.
    # For now, to keep the script runnable, we'll pass what main currently expects,
    # or comment out the main call if its signature changes drastically.
    # Current main signature: main(agent_type: str, export_frames_path_arg: Optional[str])
    # We will map the new args to the old main signature for now.
    # This will be properly fixed in the next subtask.
    agent_type_for_main = "trained" if args.agent_control == "model" else "random"
    main(
        agent_control_arg=args.agent_control,
        agent_model_path_arg=args.agent_model_path,
        enemy_model_path_arg=args.enemy_model_path,
        export_frames_path_arg=args.export_frames_path
    )
