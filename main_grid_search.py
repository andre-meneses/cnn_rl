import itertools
import os
from environment_wrapper import EnvironmentWrapper
from deep_q_agent import DeepQLearningAgent
from trainer import Trainer
from metrics import plot_metrics

def run_experiment(params):
    env = EnvironmentWrapper(params['env_name'])
    agent = DeepQLearningAgent(
        learning_rate=params['learning_rate'],
        initial_epsilon=params['initial_epsilon'],
        epsilon_decay=params['epsilon_decay'],
        final_epsilon=params['final_epsilon'],
        discount_factor=params['discount_factor'],
        batch_size=params['batch_size'],
        max_memory=params['max_memory'],
        tau=params['tau'],
        observation_space=env.observation_space,
        action_space=env.action_space,
        is_double_network=params['is_double_network'],
        hidden_layers=params['hidden_layers'],
        activation=params['activation']
    )

    trainer = Trainer(agent, env, params['nb_max_episodes'], params['test_freq'])
    trainer.train()

    hyperparams = {
        'env_name': params['env_name'],
        'learning_rate': params['learning_rate'],
        'initial_epsilon': params['initial_epsilon'],
        'discount_factor': params['discount_factor'],
        'batch_size': params['batch_size'],
        'tau': params['tau'],
        'is_double_network': params['is_double_network'],
        'hidden_layers': params['hidden_layers'],
        'activation': params['activation']
    }

    # Create a subfolder for the current environment
    results_folder = f"results_{params['env_name']}"
    os.makedirs(results_folder, exist_ok=True)

    plot_metrics(trainer.rewards, trainer.accuracies, trainer.steps, agent.training_error, hyperparams, results_folder)

def main_grid_search():
    # Environments to test
    environments = ["CartPole-v1", "MountainCar-v0"]

    # Fixed parameters
    learning_rate = 1e-4
    nb_max_episodes = 1000
    initial_epsilon = 1.0
    final_epsilon = 0.01
    discount_factor = 0.99
    max_memory = 10000
    test_freq = 50
    tau = 0.01

    # Grid search parameters
    topologies = [[16, 16],[64,64],[128, 128, 128]]
    activations = ['relu', 'leaky_relu']
    batch_sizes = [32, 64]
    is_double_networks = [True, False]

    # Generate all combinations
    combinations = list(itertools.product(environments, topologies, activations, batch_sizes, is_double_networks))

    # Run experiments for all combinations
    for env_name, hidden_layers, activation, batch_size, is_double_network in combinations:
        epsilon_decay = initial_epsilon / nb_max_episodes

        params = {
            'env_name': env_name,
            'learning_rate': learning_rate,
            'nb_max_episodes': nb_max_episodes,
            'initial_epsilon': initial_epsilon,
            'epsilon_decay': epsilon_decay,
            'final_epsilon': final_epsilon,
            'discount_factor': discount_factor,
            'batch_size': batch_size,
            'max_memory': max_memory,
            'test_freq': test_freq,
            'tau': tau,
            'is_double_network': is_double_network,
            'hidden_layers': hidden_layers,
            'activation': activation
        }

        print(f"Running experiment with: Environment: {env_name}, Hidden layers: {hidden_layers}, "
              f"Activation: {activation}, Batch size: {batch_size}, "
              f"Double DQN: {is_double_network}")

        run_experiment(params)

if __name__ == "__main__":
    main_grid_search()
