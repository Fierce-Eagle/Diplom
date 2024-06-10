import torch.nn as nn


class Perceptron_fc1(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super(Perceptron_fc1, self).__init__()
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, num_classes),
        )

    def forward(self, x):
        nn.LayerNorm(self.input_size)
        x = self.fc(x)
        return nn.functional.log_softmax(x, dim=1)


class Perceptron_fc2(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super(Perceptron_fc2, self).__init__()
        self.input_size = input_size
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 600),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.LeakyReLU(),
            nn.Linear(300, num_classes),
        )

    def forward(self, x):
        nn.LayerNorm(self.input_size)
        x = self.fc(x)
        return nn.functional.log_softmax(x, dim=1)


class Perceptron_fc3(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super(Perceptron_fc3, self).__init__()
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, num_classes),
        )

    def forward(self, x):
        x = self.fc(x)
        return nn.functional.log_softmax(x, dim=1)