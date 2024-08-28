import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
from neural_network import LinearNetwork

class DeepQLearningAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        batch_size: int,
        max_memory: int,
        tau: float,
        observation_space: np.ndarray,
        action_space: int,
        is_double_network: bool,
        hidden_layers: list,
        activation: str
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.tau = tau
        self.observation_space = observation_space
        self.action_space = action_space
        self.is_double_network = is_double_network
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.training_error = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayBuffer(self.max_memory, self.observation_space.shape[0])

        self.policy_dqn = LinearNetwork(
            self.observation_space.shape[0],
            self.action_space.n,
            self.hidden_layers,
            self.activation
        ).to(self.device)

        if self.is_double_network:
            self.target_dqn = LinearNetwork(
                self.observation_space.shape[0], 
                self.action_space.n, 
                self.hidden_layers,
                self.activation
            ).to(self.device)
            self.target_dqn.eval()
            for target_param, policy_param in zip(self.target_dqn.parameters(), self.policy_dqn.parameters()):
                target_param.data.copy_(policy_param)
            print(f"Double Deep Q-learning agent started with PyTorch. Network topology: {self.hidden_layers}, Activation: {self.activation}")
        else:
            print(f"Deep Q-learning agent started with PyTorch. Network topology: {self.hidden_layers}, Activation: {self.activation}")

        self.optimizer = optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate)
    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action = self.policy_dqn.forward(state).argmax(dim=-1)
            return action.cpu().numpy()

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return self.act(state)

    def remember(self, state, action, reward, new_state, is_terminal):
        self.memory.update(state, action, reward, new_state, is_terminal)

    def update(self):
        if self.batch_size * 10 > self.memory.size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.as_tensor(states).to(self.device)
        actions = torch.as_tensor(actions).to(self.device).unsqueeze(-1)
        rewards = torch.as_tensor(rewards).to(self.device).unsqueeze(-1)
        next_states = torch.as_tensor(next_states).to(self.device)
        dones = torch.as_tensor(dones).to(self.device).unsqueeze(-1)

        Q1 = self.policy_dqn.forward(states).gather(-1, actions.long())

        with torch.no_grad():
            if self.is_double_network:
                Q2 = self.target_dqn.forward(next_states).max(dim=-1, keepdim=True)[0]
            else:
                Q2 = self.policy_dqn.forward(next_states).max(dim=-1, keepdim=True)[0]
            target = (rewards + (1 - dones) * self.discount_factor * Q2).to(self.device)

        loss = F.mse_loss(Q1, target)
        self.training_error.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.is_double_network:
            for target_param, policy_param in zip(self.target_dqn.parameters(), self.policy_dqn.parameters()):
                target_param.data.copy_(self.tau * policy_param + (1 - self.tau) * target_param)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
