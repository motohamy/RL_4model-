from gym_env import GymEnvironment
from dqn_agent import DQNAgent
from ddpg_agent import DDPGAgent
from ppo_agent import PPOAgent
from sac_agent import SACAgent
from training_manager import TrainingManager
import os
import torch
import argparse

def create_agent(agent_type, state_dim, action_dim, is_discrete):
    """Create the specified type of agent"""
    if is_discrete:
        if agent_type != "dqn":
            raise ValueError(f"{agent_type} does not support discrete actions. Use DQN instead.")
        return DQNAgent(state_dim, action_dim)
    else:
        if agent_type == "ddpg":
            return DDPGAgent(state_dim, action_dim)
        elif agent_type == "ppo":
            return PPOAgent(state_dim, action_dim)
        elif agent_type == "sac":
            return SACAgent(state_dim, action_dim)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

def main():
    parser = argparse.ArgumentParser(description='Train RL agents in Gym environments')
    parser.add_argument('--agent', type=str, default='sac', 
                      choices=['dqn', 'ddpg', 'ppo', 'sac'],
                      help='RL algorithm to use')
    parser.add_argument('--env', type=str, default='Pendulum-v1',
                      help='Gym environment name')
    parser.add_argument('--episodes', type=int, default=1000,
                      help='Number of episodes to train')
    parser.add_argument('--max_steps', type=int, default=1000,
                      help='Maximum steps per episode')
    args = parser.parse_args()

    try:
        # Initialize environment
        print(f"Initializing {args.env} environment...")
        env = GymEnvironment(args.env)
        
        # Create agent based on action space
        print(f"Creating {args.agent.upper()} agent...")
        agent = create_agent(args.agent, env.state_dim, env.action_dim, env.is_discrete)
        
        # Initialize trainer
        trainer = TrainingManager(env, agent, args.agent)
        
        # Train agent
        print(f"Starting training for {args.episodes} episodes...")
        trainer.train(args.episodes, args.max_steps)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    main()