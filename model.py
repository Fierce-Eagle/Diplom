import torch
import torch.nn as nn


class Perceptron(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Perceptron, self).__init__()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),  # [3000, 512]
            nn.ReLU(),
            nn.Linear(512, 256),  # [512, 1024]
            nn.ReLU(),
            nn.Linear(256, 64),  # [1024, 128]
            nn.ReLU(),
            nn.Linear(64, num_classes),  # [128, 8]
        )

    def forward(self, x):
        x = self.fc(x)
        return nn.functional.log_softmax(x, dim=1)


class Regression(nn.Module):
    def __init__(self, input_size, output_size):
        super(Regression, self).__init__()
        self.layer = torch.nn.Linear(input_size, output_size, bias=True)

    def forward(self, x):
        x = self.layer(x)
        return x
