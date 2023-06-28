import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import consts
from .consts import EventType
from .params import EPV
from ..salary import utils


class NBA(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=24, hidden_size=64, num_layers=3, batch_first=True, dropout=0.3)
        self.linear = nn.Linear(in_features=64, out_features=4)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1]
        out = self.linear(out)
        return out


def train() -> None:
    config = EPV()

    x_train, y_train = get_xy(consts.EPV_PROCESSED_TRAIN_PATH, config=config)
    x_valid, y_valid = get_xy(consts.EPV_PROCESSED_VALID_PATH, config=config)
    print(numpy.unique(y_train, return_counts=True), numpy.unique(y_valid, return_counts=True))

    x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
    x_valid, y_valid = torch.from_numpy(x_valid), torch.from_numpy(y_valid)
    print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)

    train_loader = DataLoader(TensorDataset(x_train, y_train), shuffle=False, batch_size=64, pin_memory=True)
    valid_loader = DataLoader(TensorDataset(x_valid, y_valid), shuffle=False, batch_size=128, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NBA().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss(reduction="mean").to(device)

    n_epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(n_epochs):
        print("epoch", epoch + 1)

        model.train()
        train_loss, train_metric = 0.0, 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.shape[0]
            train_metric += (y_batch == torch.argmax(y_pred, dim=1)).sum()

        train_loss /= len(train_loader.dataset)
        train_metric /= len(train_loader.dataset)

        scheduler.step()

        model.eval()
        valid_loss, valid_metric = 0.0, 0.0
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                y_pred = model(x_batch)
                mask = y_batch == torch.argmax(y_pred, dim=1)

                loss = criterion(y_pred, y_batch)
                valid_loss += loss.item() * x_batch.shape[0]
                valid_metric += mask.sum()

                # if y_batch[~mask].nelement() > 0:
                #    print(numpy.unique(y_batch[~mask].detach().cpu().numpy(), return_counts=True),
                #          (y_batch != 0).int().sum().item())

            valid_loss /= len(valid_loader.dataset)
            valid_metric /= len(valid_loader.dataset)

        print(f"train: loss={train_loss}, acc={train_metric}; valid: loss={valid_loss}, acc={valid_metric}")


def get_xy(path: str, config: EPV) -> (numpy.array, numpy.array):
    data = utils.load_data(path, numeric32=True)

    length, r = 128, 16
    x, y = [], []
    for (_, _, event_type), moments in data.groupby(["game_id", "event_id", "event_type"]):
        moments = moments[config.cols]

        t = length + r

        if t > len(moments):
            continue

        while t <= len(moments):
            x.append(moments[t - (length + r):(t - r)].values)
            y.append(0)

            if t < len(moments):
                t = min(t + 64, len(moments))
                continue

            if event_type in [EventType.FIELD_GOAL_MADE, EventType.FIELD_GOAL_MISS]:
                y[-1] = 1
            elif event_type == EventType.FOUL:
                y[-1] = 2
            elif event_type in [EventType.TURNOVER, EventType.VIOLATION]:
                y[-1] = 3
            else:
                raise Exception()
            break

    x = numpy.array(x, copy=False)
    y = numpy.array(y, dtype=numpy.int64, copy=False)
    return x, y


if __name__ == "__main__":
    train()
