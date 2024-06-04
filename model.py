import torch
import torch.nn as nn


class Perceptron(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super(Perceptron, self).__init__()
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 600), # 1200
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, num_classes),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 1200),  # 1200
            nn.ReLU(),
            nn.Linear(1200, 900),
            nn.ReLU(),
            nn.Linear(900, 750),
            nn.ReLU(),
            nn.Linear(750, 600),
            nn.ReLU(),
            nn.Linear(600, 450),
            nn.ReLU(),
            nn.Linear(450, 300),
            nn.ReLU(),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 30),
            nn.ReLU(),
            nn.Linear(30, num_classes),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, num_classes),  # 1200
        )

        self.fc3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 1000),  # 1200
            nn.Sigmoid(),
            nn.Linear(1000, 800),
            nn.Sigmoid(),
            nn.Linear(800, 600),
            nn.Sigmoid(),
            nn.Linear(600, 400),
            nn.Sigmoid(),
            nn.Linear(400, 300),
            nn.Sigmoid(),
            nn.Linear(300, 200),
            nn.Sigmoid(),
            nn.Linear(200, 150),
            nn.Sigmoid(),
            nn.Linear(150, 75),
            nn.Sigmoid(),
            nn.Linear(75, 30),
            nn.Sigmoid(),
            nn.Linear(30, num_classes)
        )

        self.fc4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 400),  # 1200
            nn.ReLU(),
            nn.LayerNorm(400),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.LayerNorm(200),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.LayerNorm(100),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.LayerNorm(50),
            nn.Linear(50, num_classes)
        )

    def forward(self, x):
        nn.LayerNorm(self.input_size)
        x = self.fc4(x)
        return nn.functional.log_softmax(x, dim=1)


class Regression(nn.Module):
    def __init__(self, input_size):
        super(Regression, self).__init__()
        self.layer = torch.nn.Linear(input_size, 1, bias=True)

    def forward(self, x):
        y_pred = self.layer(x)
        return y_pred

