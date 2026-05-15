import torch
import torch.nn as nn

class LambdaPolicy(nn.Module):
    def __init__(self, state_dim, K):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, K)
        )

    def forward(self, state):
        logits = self.net(state)
        lambdas = torch.softmax(logits, dim=-1)
        return lambdas
