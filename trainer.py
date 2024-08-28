from tqdm import tqdm

class Trainer:
    def __init__(self, agent, env, nb_max_episodes, test_freq):
        self.agent = agent
        self.env = env
        self.nb_max_episodes = nb_max_episodes
        self.test_freq = test_freq
        self.total_steps = 0
        self.rewards = []
        self.steps = []
        self.accuracies = []
        self.mean_rewards = []

    def train(self):
        for episode_count in tqdm(range(self.nb_max_episodes)):
            state, _ = self.env.reset()
            is_terminal = False
            total_reward = 0
            episode_step = 0

            while not is_terminal:
                episode_step += 1
                action = self.agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                is_terminal = terminated or truncated
                self.agent.remember(state, action, reward, next_state, is_terminal)
                self.agent.update()
                state = next_state
                self.total_steps += 1
                total_reward += reward

            self.agent.decay_epsilon()
            self.mean_rewards.append(total_reward)
            
            if (episode_count + 1) % self.test_freq == 0:
                accur = self.test_accuracy(25)
                self.accuracies.append(accur)
                print(f'step: {self.total_steps}, episode: {episode_count+1}, '
                      f'training reward mean: {sum(self.mean_rewards)/self.test_freq:.2f}, '
                      f'test reward mean: {accur:.2f}, '
                      f'random move probability: {self.agent.epsilon:.2f}')
                self.mean_rewards.clear()

            self.rewards.append(total_reward)
            self.steps.append(episode_step)

    def test_accuracy(self, num_episodes=100):
        total_reward = 0.0
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            is_terminal = False
            while not is_terminal:
                action = self.agent.act(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                is_terminal = terminated or truncated
                total_reward += reward
                if is_terminal:
                    break
        return total_reward / num_episodes
