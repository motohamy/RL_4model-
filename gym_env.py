import gymnasium as gym
import numpy as np
import torch

class GymEnvironment:
    def __init__(self, env_name='Pendulum-v1'):
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        
        # Handle both continuous and discrete action spaces
        if isinstance(self.env.action_space, gym.spaces.Box):
            self.action_dim = self.env.action_space.shape[0]
            self.is_discrete = False
            self.action_low = self.env.action_space.low
            self.action_high = self.env.action_space.high
        else:
            self.action_dim = self.env.action_space.n
            self.is_discrete = True
    
    def reset(self):
        """Reset environment and return initial state"""
        state, _ = self.env.reset()
        return torch.FloatTensor(state)
    
    def step(self, action):
        """Execute action and return (next_state, reward, done)"""
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
            
        if not self.is_discrete:
            # Handle scalar and array actions
            if np.isscalar(action):
                action = np.array([action])
            
            # Scale action from [-1, 1] to actual action space
            scaled_action = self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)
            scaled_action = np.clip(scaled_action, self.action_low, self.action_high)
        else:
            scaled_action = action
            
        next_state, reward, terminated, truncated, _ = self.env.step(scaled_action)
        done = terminated or truncated
        
        return torch.FloatTensor(next_state), float(reward), done
    
    def render(self):
        """Render the environment"""
        return self.env.render()
    
    def close(self):
        """Close the environment"""
        self.env.close()