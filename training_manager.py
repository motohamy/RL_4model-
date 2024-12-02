import os
import json
import numpy as np
from datetime import datetime
import torch

class TrainingManager:
    def __init__(self, env, agent, agent_type, log_dir='logs'):
        self.env = env
        self.agent = agent
        self.agent_type = agent_type
        self.log_dir = os.path.join(log_dir, agent_type, datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.rewards_history = []
        self.losses_history = []
        
        # Save training configuration
        self.save_config()
    
    def save_config(self):
        config = {
            'agent_type': self.agent_type,
            'state_dim': int(self.env.state_dim),  # Convert to int
            'action_dim': int(self.env.action_dim),  # Convert to int
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    
    def train(self, num_episodes, max_steps=1000):
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_losses = []
            
            for step in range(max_steps):
                # Handle different action selection methods
                if self.agent_type == "ppo":
                    action, log_prob, value = self.agent.select_action(state)
                else:
                    action = self.agent.select_action(state)
                
                next_state, reward, done = self.env.step(action)
                
                # Store transitions based on agent type
                if self.agent_type == "ppo":
                    self.agent.store_transition(state, action, reward, next_state, done, log_prob, value)
                else:
                    self.agent.store_transition(state, action, reward, next_state, done)
                
                # Train the agent
                if self.agent_type == "ppo":
                    if (step + 1) % self.agent.batch_size == 0:
                        loss = self.agent.train()
                        if loss:
                            episode_losses.append(loss)
                else:
                    loss = self.agent.train()
                    if loss:
                        episode_losses.append(loss)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Record metrics
            self.rewards_history.append(episode_reward)
            if episode_losses:
                # Handle different loss formats
                if isinstance(episode_losses[0], dict):
                    avg_loss = np.mean([loss['actor_loss'] for loss in episode_losses])
                else:
                    avg_loss = np.mean(episode_losses)
                self.losses_history.append(avg_loss)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.rewards_history[-10:])
                print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}")
                self._save_checkpoint(episode)
                self._save_metrics()
    
    def _save_checkpoint(self, episode):
        checkpoint_path = os.path.join(self.log_dir, f'checkpoint_episode_{episode}.pt')
        self.agent.save(checkpoint_path)
    
    def _save_metrics(self):
        metrics = {
            'rewards': self.rewards_history,
            'losses': self.losses_history
        }
        metrics_path = os.path.join(self.log_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)