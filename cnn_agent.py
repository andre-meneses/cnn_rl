import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from cnn_network import ConvolutionalNetwork

class CNNReplayBuffer:
    def __init__(self, max_size, batch_size, device):
        self.memory = []
        self.max_size = max_size
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.max_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])
        return (torch.stack(states).to(self.device),
                torch.LongTensor(actions).to(self.device),
                torch.FloatTensor(rewards).to(self.device),
                torch.stack(next_states).to(self.device),
                torch.FloatTensor(dones).to(self.device))

    def __len__(self):
        return len(self.memory)

class CNNDeepQLearningAgent:
    def __init__(
        self,
        learning_rate,
        initial_epsilon,
        epsilon_decay,
        final_epsilon,
        discount_factor,
        batch_size,
        max_memory,
        tau,
        input_shape,
        action_space,
        conv_layers,
        is_double_network
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.batch_size = batch_size
        self.tau = tau
        self.input_shape = input_shape
        self.action_space = action_space
        self.is_double_network = is_double_network
        self.training_error = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = CNNReplayBuffer(max_memory, batch_size, self.device)

        self.policy_dqn = ConvolutionalNetwork(input_shape, action_space.n, conv_layers).to(self.device)

        if self.is_double_network:
            self.target_dqn = ConvolutionalNetwork(input_shape, action_space.n, conv_layers).to(self.device)
            self.target_dqn.eval()
            for target_param, policy_param in zip(self.target_dqn.parameters(), self.policy_dqn.parameters()):
                target_param.data.copy_(policy_param.data)
            print("Double Deep Q-learning agent with CNN started")
        else:
            print("Deep Q-learning agent with CNN started")

        self.optimizer = optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate)

    def choose_action(self, state, is_evaluating=False):
        if not is_evaluating and np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            state = state.unsqueeze(0).to(self.device)
            state = state.view(state.size(0), -1, state.size(-2), state.size(-1))
            self.policy_dqn.eval()
            with torch.no_grad():
                action_values = self.policy_dqn.forward(state)
            self.policy_dqn.train()
            return np.argmax(action_values.cpu().data.numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()

        # Reshape states and next_states
        states = states.view(states.size(0), -1, states.size(-2), states.size(-1))
        next_states = next_states.view(next_states.size(0), -1, next_states.size(-2), next_states.size(-1))

        Q1 = self.policy_dqn.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        if self.is_double_network:
            Q2 = self.target_dqn.forward(next_states).detach().max(1)[0]
        else:
            Q2 = self.policy_dqn.forward(next_states).detach().max(1)[0]

        target = rewards + (1 - dones) * self.discount_factor * Q2

        loss = F.mse_loss(Q1, target)
        self.training_error.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.is_double_network:
            for target_param, policy_param in zip(self.target_dqn.parameters(), self.policy_dqn.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
