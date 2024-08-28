# Deep Q-Learning for Atari and Classic Control Environments

This project implements Deep Q-Learning agents to solve both Atari games and classic control problems from OpenAI's Gymnasium. It includes grid search functionalities to optimize hyperparameters across different network architectures and learning strategies.

## Project Overview

The project consists of two main parts:

### 1. CNN-based Deep Q-Learning for Atari Games

- CNN-based Deep Q-Learning agent (`cnn_agent.py`)
- Customizable convolutional neural network architecture (`cnn_network.py`)
- Grid search script for Atari environments (`grid_search_cnn.py`)

This part is designed to work with image-based Atari environments, specifically "SpaceInvaders-v5" and "Galaxian-v5".

### 2. Deep Q-Learning for Classic Control Problems

- Deep Q-Learning agent for classic control problems (`deep_q_agent.py`)
- Neural network for classic control (`neural_network.py`)
- Grid search script for classic control environments (`main_grid_search.py`)

This part works with classic control environments like "CartPole-v1" and "MountainCar-v0".

## Requirements

- Python 3.7+
- PyTorch
- Gymnasium with Atari and classic control environments
- OpenCV (for Atari environments)
- NumPy
- Matplotlib
- tqdm

You can install the required packages using:

```
pip install torch gymnasium[atari,classic_control] opencv-python numpy matplotlib tqdm
```

## Running the Grid Searches

### Atari Environments

#### Local Execution

To run the grid search for Atari environments locally:

```
python grid_search_cnn.py
```

#### Cluster Execution with SLURM

To run on a multi-GPU cluster using SLURM:

```
sbatch run_grid_search_slurm.py
```

### Classic Control Environments

To run the grid search for classic control environments:

```
python main_grid_search.py
```

## Grid Search Parameters

### Atari Environments

The grid search explores combinations of:

- Convolutional layer configurations
- Frame stack sizes
- Memory buffer sizes
- Batch sizes
- Single vs Double DQN

### Classic Control Environments

The grid search explores combinations of:

- Neural network topologies
- Activation functions (ReLU or Leaky ReLU)
- Batch sizes
- Single vs Double DQN

You can modify these parameters in the respective grid search scripts.

## Results

### Atari Environments

Results are saved as PDF files in separate directories for each game:

- `results_SpaceInvaders/`
- `results_Galaxian/`

### Classic Control Environments

Results are saved as PDF files in separate directories for each environment:

- `results_CartPole-v1/`
- `results_MountainCar-v0/`

Each PDF includes plots of episode rewards, test accuracies, and training errors.

## Customization

- To add new environments, modify the `environments` list in the respective grid search scripts.
- To change the neural network architecture, modify the `ConvolutionalNetwork` class in `cnn_network.py` for Atari environments, or the `LinearNetwork` class in `neural_network.py` for classic control environments.
- To adjust the learning process, modify the `CNNDeepQLearningAgent` class in `cnn_agent.py` for Atari environments, or the `DeepQLearningAgent` class in `deep_q_agent.py` for classic control environments.


## Project Structure

```
.
├── cnn_agent.py
├── cnn_network.py
├── deep_q_agent.py
├── grid_search_cnn.py
├── main_grid_search.py
├── neural_network.py
├── README.md
└── gpu_cnn
```

Each script contains detailed comments explaining its functionality and usage.
