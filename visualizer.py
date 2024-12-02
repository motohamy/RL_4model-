import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime

class MetricsVisualizer:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        
    def load_metrics(self, agent_type, run_timestamp=None):
        """Load metrics from a specific agent run"""
        if run_timestamp:
            path = os.path.join(self.log_dir, agent_type, run_timestamp, 'metrics.json')
        else:
            # Get most recent run if timestamp not specified
            agent_dir = os.path.join(self.log_dir, agent_type)
            runs = sorted(os.listdir(agent_dir))
            if not runs:
                raise ValueError(f"No runs found for {agent_type}")
            path = os.path.join(agent_dir, runs[-1], 'metrics.json')
            
        with open(path, 'r') as f:
            return json.load(f)
    
    def plot_training_curves(self, agent_types=['dqn', 'ddpg', 'ppo', 'sac']):
        """Plot comparison of training curves for different agents"""
        plt.figure(figsize=(15, 5))
        
        # Plot rewards
        plt.subplot(1, 2, 1)
        for agent_type in agent_types:
            try:
                metrics = self.load_metrics(agent_type)
                rewards = metrics['rewards']
                plt.plot(rewards, label=agent_type.upper())
            except Exception as e:
                print(f"Could not load metrics for {agent_type}: {e}")
        
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        
        # Plot losses
        plt.subplot(1, 2, 2)
        for agent_type in agent_types:
            try:
                metrics = self.load_metrics(agent_type)
                losses = metrics['losses']
                plt.plot(losses, label=agent_type.upper())
            except Exception as e:
                print(f"Could not load metrics for {agent_type}: {e}")
        
        plt.title('Training Losses')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_agent_performance(self, agent_type):
        """Plot detailed performance metrics for a specific agent"""
        metrics = self.load_metrics(agent_type)
        
        plt.figure(figsize=(15, 10))
        
        # Rewards
        plt.subplot(2, 2, 1)
        rewards = metrics['rewards']
        plt.plot(rewards)
        plt.title(f'{agent_type.upper()} Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Rolling average rewards
        plt.subplot(2, 2, 2)
        window = 100
        rolling_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(rolling_rewards)
        plt.title(f'{window}-Episode Rolling Average Reward')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        
        # Losses
        plt.subplot(2, 2, 3)
        losses = metrics['losses']
        plt.plot(losses)
        plt.title(f'{agent_type.upper()} Losses')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        # Reward distribution
        plt.subplot(2, 2, 4)
        plt.hist(rewards, bins=50)
        plt.title('Reward Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
    def save_summary(self, agent_type):
        """Save summary statistics for an agent"""
        metrics = self.load_metrics(agent_type)
        rewards = metrics['rewards']
        
        summary = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'final_100_episodes_mean': np.mean(rewards[-100:])
        }
        
        summary_path = os.path.join(self.log_dir, agent_type, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        return summary