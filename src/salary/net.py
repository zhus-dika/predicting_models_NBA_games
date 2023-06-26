from torch import nn


class Regressor(nn.Module):
    def __init__(self, num_inputs: int, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=num_inputs, out_features=width),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(in_features=width, out_features=width),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(in_features=width, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
