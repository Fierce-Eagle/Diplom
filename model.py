import torch
import torch.nn as nn


class Perceptron(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(Perceptron, self).__init__()

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
            nn.Linear(75, 30),
            nn.ReLU(),
            nn.Linear(30, num_classes)
        )

    def forward(self, x):
        x = self.fc(x)
        return nn.functional.log_softmax(x, dim=1)


class Regression(nn.Module):
    def __init__(self, input_size):
        super(Regression, self).__init__()
        self.layer = torch.nn.Linear(input_size, 1, bias=True)

    def forward(self, x):
        y_pred = self.layer(x)
        return y_pred

