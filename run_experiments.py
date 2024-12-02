import subprocess
import time
import json
from datetime import datetime

def run_experiments():
    experiments = [
        # SAC experiments
        {"agent": "sac", "env": "Pendulum-v1", "episodes": 100},
        {"agent": "sac", "env": "LunarLanderContinuous-v3", "episodes": 100},  # Updated version
        
        # DDPG experiments
        {"agent": "ddpg", "env": "Pendulum-v1", "episodes": 100},
        {"agent": "ddpg", "env": "LunarLanderContinuous-v3", "episodes": 100},  # Updated version
        
        # PPO experiments
        {"agent": "ppo", "env": "Pendulum-v1", "episodes": 100},
        {"agent": "ppo", "env": "HalfCheetah-v4", "episodes": 100},  # Alternative to BipedalWalker
        
        # DQN experiments
        {"agent": "dqn", "env": "CartPole-v1", "episodes": 100},
        {"agent": "dqn", "env": "LunarLander-v3", "episodes": 100}  # Updated version
    ]
    
    for exp in experiments:
        try:
            print(f"\nRunning experiment: {exp}")
            cmd = f"python main.py --agent {exp['agent']} --env {exp['env']} --episodes {exp['episodes']}"
            subprocess.run(cmd, shell=True)
        except Exception as e:
            print(f"Error in experiment {exp}: {str(e)}")

if __name__ == "__main__":
    run_experiments()