import matplotlib.pyplot as plt
import numpy as np
import os

def plot_metrics(rewards, accuracies, steps, training_error, hyperparams, results_folder, rolling_length=100):
    fig, axs = plt.subplots(ncols=4, figsize=(20, 5))

    axs[0].set_title("Training Reward")
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')
    axs[0].plot(range(len(rewards)), rewards)

    axs[1].set_title("Test Reward")
    axs[1].set_xlabel('Episode (x 25)')
    axs[1].set_ylabel('Accuracy')
    axs[1].plot(range(len(accuracies)), accuracies)

    axs[2].set_title("Episode Length")
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Step')
    steps_moving_average = np.convolve(np.array(steps), np.ones(rolling_length), mode="same") / rolling_length
    axs[2].plot(range(len(steps_moving_average)), steps_moving_average)

    axs[3].set_title("Training Error")
    axs[3].set_xlabel('Step')
    axs[3].set_ylabel('Error')
    training_error_moving_average = np.convolve(np.array(training_error), np.ones(rolling_length), mode="same") / rolling_length
    axs[3].plot(range(len(training_error_moving_average)), training_error_moving_average)

    plt.tight_layout()

    # Create filename with hyperparameters
    filename = f"{hyperparams['env_name']}_DQN_lr{hyperparams['learning_rate']}_eps{hyperparams['initial_epsilon']}_" \
               f"gamma{hyperparams['discount_factor']}_batch{hyperparams['batch_size']}_" \
               f"tau{hyperparams['tau']}_{'Double' if hyperparams['is_double_network'] else 'Single'}_" \
               f"layers{'_'.join(map(str, hyperparams['hidden_layers']))}_" \
               f"{hyperparams['activation']}.pdf"

    # Save the plot in the appropriate folder
    full_path = os.path.join(results_folder, filename)
    plt.savefig(full_path)
    plt.close(fig)

    print(f"Plot saved as {full_path}")
