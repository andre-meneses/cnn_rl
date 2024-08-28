import numpy as np

class ReplayBuffer:
    def __init__(self, max_length, observation_space_n):
        self.index, self.size, self.max_length = 0, 0, max_length
        self.states = np.zeros((max_length, observation_space_n), dtype=np.float32)
        self.actions = np.zeros((max_length), dtype=np.uint8)
        self.rewards = np.zeros((max_length), dtype=np.float32)
        self.next_states = np.zeros((max_length, observation_space_n), dtype=np.float32)
        self.dones = np.zeros((max_length), dtype=np.uint8)

    def __len__(self):
        return self.size

    def update(self, state, action, reward, next_state, is_terminal):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = is_terminal
        self.index = (self.index + 1) % self.max_length
        if self.size < self.max_length:
            self.size += 1

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idxs], self.actions[idxs], self.rewards[idxs], 
                self.next_states[idxs], self.dones[idxs])
