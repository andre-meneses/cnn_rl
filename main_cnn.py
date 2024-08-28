import gym
import numpy as np
from cnn_agent import CNNDeepQLearningAgent
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def preprocess_frame(state, crop_box, output_size):
    state = state[crop_box[0]:crop_box[1], crop_box[2]:crop_box[3]]
    state = np.ascontiguousarray(state, dtype=np.float32) / 255
    return torch.from_numpy(state).resize((1, output_size, output_size))

def stack_frames(frames, state, is_new, stack_size, crop_box, output_size):
    frame = preprocess_frame(state, crop_box, output_size)
    if is_new:
        frames = np.stack([frame for _ in range(stack_size)], axis=0)
    else:
        frames = np.roll(frames, shift=-1, axis=0)
        frames[-1] = frame
    return frames

def test_accuracy(env, agent, num_episodes, stack_size, crop_box, output_size):
    total_reward = 0.0
    for _ in range(num_episodes):
        state = env.reset()[0]
        frames = stack_frames(None, state, True, stack_size, crop_box, output_size)
        done = False
        while not done:
            action = agent.choose_action(frames, is_evaluating=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frames = stack_frames(frames, next_state, False, stack_size, crop_box, output_size)
            total_reward += reward
    return total_reward / num_episodes

def main():
    env_name = "SpaceInvaders-v4"
    env = gym.make(env_name, render_mode="rgb_array")
    
    # Hyperparameters
    learning_rate = 0.0001
    initial_epsilon = 1.0
    epsilon_decay = 0.99995
    final_epsilon = 0.1
    discount_factor = 0.99
    batch_size = 32
    max_memory = 100000
    tau = 0.001
    nb_max_episodes = 1000
    test_freq = 50
    number_of_tests = 5
    
    stack_size = 4
    crop_box = (8, -12, -12, 4)
    output_size = 84
    input_shape = (stack_size, output_size, output_size)
    
    conv_layers = [
        (32, 8, 4),
        (64, 4, 2),
        (64, 3, 1)
    ]
    
    agent = CNNDeepQLearningAgent(
        learning_rate=learning_rate,
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
        batch_size=batch_size,
        max_memory=max_memory,
        tau=tau,
        input_shape=input_shape,
        action_space=env.action_space,
        conv_layers=conv_layers,
        is_double_network=True
    )

    rewards = []
    accuracies = []
    steps = []
    total_steps = 0

    for episode in tqdm(range(nb_max_episodes)):
        state = env.reset()[0]
        frames = stack_frames(None, state, True, stack_size, crop_box, output_size)
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(frames)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_frames = stack_frames(frames, next_state, False, stack_size, crop_box, output_size)
            agent.remember(frames, action, reward, next_frames, done)
            agent.update()
            frames = next_frames
            total_reward += reward
            total_steps += 1

        agent.decay_epsilon()
        rewards.append(total_reward)
        steps.append(total_steps)

        if (episode + 1) % test_freq == 0:
            accuracy = test_accuracy(env, agent, number_of_tests, stack_size, crop_box, output_size)
            accuracies.append(accuracy)
            print(f'Episode: {episode+1}, Test Reward: {accuracy:.2f}, Epsilon: {agent.epsilon:.4f}')

    env.close()

    # Plot results
    fig, axs = plt.subplots(ncols=3, figsize=(18, 5))
    axs[0].plot(rewards)
    axs[0].set_title('Episode Rewards')
    axs[1].plot(accuracies)
    axs[1].set_title('Test Accuracies')
    axs[2].plot(agent.training_error)
    axs[2].set_title('Training Error')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
