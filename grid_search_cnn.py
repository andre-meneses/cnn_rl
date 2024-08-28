import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import os
from cnn_agent import CNNDeepQLearningAgent
import cv2

def preprocess_frame(state, output_size):
    # Convert to grayscale and resize
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (output_size, output_size), interpolation=cv2.INTER_AREA)
    # Normalize
    normalized = resized / 255.0
    # Add channel dimension
    channels = np.expand_dims(normalized, axis=0)
    return torch.FloatTensor(channels)

def stack_frames(frames, state, is_new, stack_size, output_size):
    frame = preprocess_frame(state, output_size)
    if is_new:
        # For a new episode, copy the same frame stack_size times
        frames = np.repeat(frame.unsqueeze(0), stack_size, axis=0)
    else:
        # Remove the oldest frame and add the new one
        frames = torch.cat((frames[1:], frame.unsqueeze(0)), dim=0)
    return frames

def test_accuracy(env, agent, num_episodes, stack_size, output_size):
    total_reward = 0.0
    for _ in range(num_episodes):
        state, _ = env.reset()
        frames = stack_frames(None, state, True, stack_size, output_size)
        done = False
        while not done:
            action = agent.choose_action(frames, is_evaluating=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frames = stack_frames(frames, next_state, False, stack_size, output_size)
            total_reward += reward
    return total_reward / num_episodes

def run_experiment(params):
    env = gym.make(params['env_name'])

    agent = CNNDeepQLearningAgent(
        learning_rate=params['learning_rate'],
        initial_epsilon=params['initial_epsilon'],
        epsilon_decay=params['epsilon_decay'],
        final_epsilon=params['final_epsilon'],
        discount_factor=params['discount_factor'],
        batch_size=params['batch_size'],
        max_memory=params['max_memory'],
        tau=params['tau'],
        input_shape=params['input_shape'],
        action_space=env.action_space,
        conv_layers=params['conv_layers'],
        is_double_network=params['is_double_network']
    )

    rewards = []
    accuracies = []
    total_steps = 0

    for episode in tqdm(range(params['nb_max_episodes'])):
        state, _ = env.reset()
        frames = stack_frames(None, state, True, params['stack_size'], params['output_size'])
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(frames)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_frames = stack_frames(frames, next_state, False, params['stack_size'], params['output_size'])
            agent.remember(frames, action, reward, next_frames, done)
            agent.update()
            frames = next_frames
            total_reward += reward
            total_steps += 1

        agent.decay_epsilon()
        rewards.append(total_reward)

        if (episode + 1) % params['test_freq'] == 0:
            accuracy = test_accuracy(env, agent, params['number_of_tests'], params['stack_size'], params['output_size'])
            accuracies.append(accuracy)
            print(f'Episode: {episode+1}, Test Reward: {accuracy:.2f}, Epsilon: {agent.epsilon:.4f}')

    env.close()

    return rewards, accuracies, agent.training_error

def main_grid_search():
    # Environments to test
    environments = ["ALE/SpaceInvaders-v5", "ALE/Galaxian-v5"]

    # Fixed parameters
    learning_rate = 1e-4
    initial_epsilon = 1.0
    final_epsilon = 0.1
    discount_factor = 0.99
    tau = 0.01
    nb_max_episodes = 1000
    test_freq = 50
    number_of_tests = 5
    output_size = 84

    # Grid search parameters
    conv_layers_options = [
        [(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        [(16, 8, 4), (32, 4, 2), (32, 3, 1)],
        [(64, 8, 4), (128, 4, 2), (128, 3, 1)]
    ]
    stack_sizes = [4,8]
    max_memories = [100000]
    batch_sizes = [32, 64]
    is_double_networks = [True, False]

    # Generate all combinations
    combinations = list(itertools.product(environments, conv_layers_options, stack_sizes, max_memories, batch_sizes, is_double_networks))

    for env_name, conv_layers, stack_size, max_memory, batch_size, is_double_network in combinations:
        print(f"\nRunning experiment with:")
        print(f"Environment: {env_name}")
        print(f"Conv layers: {conv_layers}")
        print(f"Stack size: {stack_size}")
        print(f"Max memory: {max_memory}")
        print(f"Batch size: {batch_size}")
        print(f"Double DQN: {is_double_network}")

        input_shape = (stack_size, output_size, output_size)
        epsilon_decay = 1 / nb_max_episodes

        params = {
            'env_name': env_name,
            'learning_rate': learning_rate,
            'initial_epsilon': initial_epsilon,
            'epsilon_decay': epsilon_decay,
            'final_epsilon': final_epsilon,
            'discount_factor': discount_factor,
            'batch_size': batch_size,
            'max_memory': max_memory,
            'tau': tau,
            'input_shape': input_shape,
            'conv_layers': conv_layers,
            'is_double_network': is_double_network,
            'nb_max_episodes': nb_max_episodes,
            'test_freq': test_freq,
            'number_of_tests': number_of_tests,
            'stack_size': stack_size,
            'output_size': output_size
        }

        rewards, accuracies, training_error = run_experiment(params)

        # Save results
        save_results(params, rewards, accuracies, training_error)

def save_results(params, rewards, accuracies, training_error):
    env_name = params['env_name'].split('/')[1].split('-')[0]  # Extract game name
    results_dir = f"results_{env_name}"
    os.makedirs(results_dir, exist_ok=True)

    fig, axs = plt.subplots(ncols=3, figsize=(18, 5))

    axs[0].plot(rewards)
    axs[0].set_title('Episode Rewards')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')

    axs[1].plot(accuracies)
    axs[1].set_title('Test Accuracies')
    axs[1].set_xlabel('Test Iteration')
    axs[1].set_ylabel('Accuracy')

    axs[2].plot(training_error)
    axs[2].set_title('Training Error')
    axs[2].set_xlabel('Update Step')
    axs[2].set_ylabel('Error')

    plt.tight_layout()

    # Create filename with hyperparameters
    filename = f"{env_name}_C{'_'.join([str(c[0]) for c in params['conv_layers']])}_" \
               f"S{params['stack_size']}_M{params['max_memory']}_B{params['batch_size']}_" \
               f"{'Double' if params['is_double_network'] else 'Single'}.pdf"

    plt.savefig(os.path.join(results_dir, filename), format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"Results saved as {os.path.join(results_dir, filename)}")

if __name__ == "__main__":
    main_grid_search()
