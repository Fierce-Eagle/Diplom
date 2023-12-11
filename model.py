import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        self.model = nn.Sequential(

        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * 3, input_size),
            nn.Tanh(),
            nn.Linear(input_size, input_size // 5),
            nn.Sigmoid(),
            nn.Linear(input_size // 5, input_size // 10),
            nn.Tanh(),
            nn.Linear(input_size // 10, num_classes),
            nn.Softmax()
        )

    def forward(self, x):

        x = self.fc(x)
        return x