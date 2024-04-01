

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import random

class LinearDeepQNetwork(nn.Module):
    def _init_(self, lr, n_actions, input_dims):
        super(LinearDeepQNetwork, self)._init_()
        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)
        return actions

class Agent():
    def _init_(self, input_dims, n_actions, lr, gamma=0.99,
                 epsilon=1.0, eps_dec=1e-5, eps_min=0.01):
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]
        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            state1 = T.tensor(state, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state1)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        self.Q.optimizer.zero_grad()
        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.Q.device)
        q_pred = self.Q.forward(states)[actions]
        q_next = self.Q.forward(states_).max()
        q_target = reward + self.gamma*q_next
        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()

def get_data_from_file(file_path):
    data = np.genfromtxt(file_path, delimiter=',')
    data = data[~np.isnan(data).any(axis=1)]
    return data

def train_agent(data, agent, epochs=100):
    for epoch in range(epochs):
        total_reward = 0
        for sample in data:
            state = sample[:-1]  # Assuming last value is the action
            action = int(sample[-1])  # Assuming action is the last value
            next_state = state  # Example: No state transition
            
            # Get the action chosen by the agent
            chosen_action = agent.choose_action(state)
            
            # Update agent with the new reward
            reward = calculate_reward(state, chosen_action, next_state)
            agent.learn(state, chosen_action, reward, next_state)
            
            # Accumulate total reward for the epoch
            total_reward += reward
        
        # Print total reward for the epoch
        print(f"Epoch {epoch + 1}, Total Reward: {total_reward}, Chosen Action: {chosen_action}")


def calculate_reward(state, action, next_state):
    # Implement the logic to calculate rewards based on the current state and action
    # Example: Return a predefined reward based on the action
    return 1 if action == 1 else 0


def main():
    file_path = "data.txt"  # Path to your data file
    data = get_data_from_file(file_path)

    input_dims = data.shape[1] - 1  # Assuming last column is action
    n_actions = len(data)  # Number of unique actions
    lr = 0.001  # Learning rate

    agent = Agent(input_dims, n_actions, lr)

    train_agent(data, agent)

if __name__ == "__main_":
    main()
