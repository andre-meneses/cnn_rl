import torch
import torch.nn as nn
import torch.autograd as autograd

class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_shape, action_space_n, conv_layers):
        super(ConvolutionalNetwork, self).__init__()
        self.input_shape = input_shape
        self.action_space_n = action_space_n

        self.layers = nn.Sequential()
        in_channels = input_shape[0]
        for i, (out_channels, kernel_size, stride) in enumerate(conv_layers):
            self.layers.add_module(f'conv{i+1}', nn.Conv2d(in_channels, out_channels, kernel_size, stride))
            self.layers.add_module(f'relu{i+1}', nn.ReLU())
            in_channels = out_channels

        print("Convolutional neural network started")
        print("Shape =", self.input_shape)

        linear_input_size = self.layers_size()
        self.fc = nn.Linear(linear_input_size, self.action_space_n)

    def layers_size(self):
        return self.layers(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def forward(self, state):
        state = self.layers(state)
        state = state.view(state.size(0), -1)
        return self.fc(state)
