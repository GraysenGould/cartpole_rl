import torch
from torch import nn



class QNeuralNetwork (nn.Module):
    def __init__ (self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward (self, state):
        logits = self.model(state)
        return logits



    