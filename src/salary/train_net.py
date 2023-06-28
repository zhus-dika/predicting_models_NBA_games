import click
import numpy
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import mlflow

from . import consts, utils
from .net import Regressor


@click.command("train_net")
@click.option("--in-train-path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.NET_PROCESSED_TRAIN_PATH)
@click.option("--in-target-preprocessor-path", type=click.Path(exists=True, dir_okay=False, readable=True),
              default=consts.NET_TARGET_PREPROCESSOR_PATH)
@click.option("--out-model-path", type=click.Path(dir_okay=False, writable=True), default=consts.NET_MODEL_PATH)
@click.option("--out-params-path", type=click.Path(dir_okay=False, writable=True), default=consts.NET_TRAIN_PARAMS_PATH)
@click.option("--target", type=str, required=True)
@click.option("--metric-name", type=str, required=True)
@click.option("--valid-size", type=click.FloatRange(min=0.0, max=1.0), required=True)
@click.option("--batch-size", type=int, required=True)
@click.option("--width", type=int, required=True)
@click.option("--dropout", type=click.FloatRange(min=0.0, max=1.0), required=True)
@click.option("--n-epochs", type=int, required=True)
@click.option("--seed", type=int, required=True)
def train_net(in_train_path: str, in_target_preprocessor_path: str, out_model_path: str, out_params_path: str,
              target: str, metric_name: str, valid_size: float, batch_size: int, width: int, dropout: float,
              n_epochs: int, seed: int) -> None:
    target_preprocessor = utils.load_preprocessor(in_target_preprocessor_path)

    def inverse_target(t: torch.Tensor) -> numpy.array:
        return target_preprocessor.inverse_transform(t.detach().cpu().numpy())

    x, y = utils.load_features_target(in_train_path, target=target)
    x = x.values
    y = target_preprocessor.transform(y.values.reshape(-1, 1))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=valid_size, shuffle=True, random_state=seed)
    x_valid = torch.tensor(x_valid, dtype=torch.float32, device=device)
    y_valid = torch.tensor(y_valid, dtype=torch.float32, device=device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

    torch.manual_seed(seed)
    numpy.random.seed(seed)

    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32, device=device),
                                  torch.tensor(y_train, dtype=torch.float32, device=device))

    pin_memory = train_dataset.tensors[0].get_device() < 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

    regressor = Regressor(num_inputs=x.shape[1], width=width, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(regressor.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction="mean").to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    mlflow.set_tracking_uri(uri=consts.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("salary_net")

    with mlflow.start_run(run_name=out_model_path):
        for epoch in range(1, n_epochs + 1):
            regressor.train()
            train_loss, train_metric = 0.0, 0.0

            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = regressor(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    metrics = utils.eval_metrics(inverse_target(y_batch), inverse_target(pred), names=[metric_name])
                    train_loss += loss.item() * x_batch.shape[0]
                    train_metric += metrics[metric_name] * x_batch.shape[0]

            train_loss /= len(train_loader.dataset)
            train_metric /= len(train_loader.dataset)

            scheduler.step()

            regressor.eval()
            with torch.no_grad():
                pred = regressor(x_valid)
                valid_loss = criterion(pred, y_valid)
                valid_metrics = utils.eval_metrics(inverse_target(y_valid), inverse_target(pred), names=[metric_name])

            metrics = {
                "train_loss": train_loss,
                f"train_{metric_name}": train_metric,
                "valid_loss": valid_loss.item(),
                f"valid_{metric_name}": valid_metrics[metric_name]
            }
            mlflow.log_metrics(metrics, step=epoch)

        utils.make_dir(utils.get_dir_path(out_model_path))
        torch.save(regressor.state_dict(), out_model_path)

        params = {
            "target": target,
            "metric_name": metric_name,
            "valid_size": valid_size,
            "batch_size": batch_size,
            "width": width,
            "dropout": dropout,
            "n_epochs": n_epochs,
            "seed": seed,
            "num_inputs": x.shape[1]
        }

        utils.save_params(params, path=out_params_path)
        mlflow.log_params(params)


if __name__ == "__main__":
    train_net()
