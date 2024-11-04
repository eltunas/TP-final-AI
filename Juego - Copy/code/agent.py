import torch
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim
from dqn_model import DQN

class Agent:
    def __init__(self, action_space, state_shape):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(action_space).to(self.device)
        self.target_model = DQN(action_space).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.memory = []  # Experiencias para memoria de repetición
        self.gamma = 0.99  # Factor de descuento
        self.epsilon = 1.0  # Tasa de exploración inicial
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1, 2, 3])  # Acción aleatoria
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def store_transition(self, transition):
        self.memory.append(transition)
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def train_step(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        # Sample a batch from memory
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Q values for current state
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values for next state
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Loss calculation and optimization
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reduce exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay