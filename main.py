from environment_wrapper import EnvironmentWrapper
from deep_q_agent import DeepQLearningAgent
from trainer import Trainer
from metrics import plot_metrics

def main():
    # Parameters
    env_name = "CartPole-v1"
    learning_rate = 1e-4
    nb_max_episodes = 400
    initial_epsilon = 1.0
    epsilon_decay = initial_epsilon / nb_max_episodes
    final_epsilon = 0.1
    discount_factor = 0.99
    batch_size = 64
    max_memory = 10000
    test_freq = 50
    tau = 0.005
    is_double_network = True
    hidden_layers = [128, 128, 128]
    activation = 'relu'  # Can be 'relu' or 'leaky_relu'

    # Initialize environment and agent
    env = EnvironmentWrapper(env_name)
    agent = DeepQLearningAgent(
        learning_rate=learning_rate,
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
        batch_size=batch_size,
        max_memory=max_memory,
        tau=tau,
        observation_space=env.observation_space,
        action_space=env.action_space,
        is_double_network=is_double_network,
        hidden_layers=hidden_layers,
        activation=activation
    )

    # Initialize trainer
    trainer = Trainer(agent, env, nb_max_episodes, test_freq)

    # Train the agent
    trainer.train()

    # Collect hyperparameters for plot filename
    hyperparams = {
        'learning_rate': learning_rate,
        'initial_epsilon': initial_epsilon,
        'discount_factor': discount_factor,
        'batch_size': batch_size,
        'tau': tau,
        'is_double_network': is_double_network,
        'hidden_layers': hidden_layers,
        'activation': activation
    }

    # Plot metrics
    plot_metrics(trainer.rewards, trainer.accuracies, trainer.steps, agent.training_error, hyperparams)

if __name__ == "__main__":
    main()
