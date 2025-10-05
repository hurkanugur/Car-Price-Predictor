import torch
from torch import nn
import config

class CarPricePredictionModel(nn.Module):
    def __init__(self, input_dim, device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1)
        )
        self.net.apply(self.init_weights)
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.net(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def save(self):
        torch.save(self.state_dict(), config.MODEL_PATH)
        print(f"• Model saved to {config.MODEL_PATH}")

    def load(self):
        self.load_state_dict(torch.load(config.MODEL_PATH, weights_only=True, map_location=self.device))
        self.to(self.device)
        print(f"• Model loaded from {config.MODEL_PATH}")
