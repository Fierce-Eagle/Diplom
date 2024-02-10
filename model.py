import torch
import torch.nn as nn


class Perceptron(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Perceptron, self).__init__()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 1024),  # [3000, 512]
            nn.Tanh(),
            nn.Linear(512, 1024),  # [512, 1024]
            nn.Sigmoid(),
            nn.Linear(1024, 128),  # [1024, 128]
            nn.Sigmoid(),
            nn.Linear(128, num_classes),  # [128, 8]
            nn.Softmax()
        )

    def forward(self, x):
        x = self.fc(x)
        return x

