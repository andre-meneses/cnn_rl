import torch
import torch.nn as nn

class LinearNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[128, 128], activation='relu'):
        super(LinearNetwork, self).__init__()

        # Set activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError("Activation must be 'relu' or 'leaky_relu'")

        # Create a list to hold all layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(self.activation)

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(self.activation)

        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_dim))

        # Combine all layers into a sequential model
        self.layers = nn.Sequential(*layers)

    def forward(self, state):
        return self.layers(state)
