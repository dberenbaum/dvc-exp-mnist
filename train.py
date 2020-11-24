"""Model training and evaluation."""
import os
import click
import torch
import torch.nn.functional as F
import torchvision
import yaml
from dvc.api import make_checkpoint


def transform(dataset):
    """Get inputs and targets from dataset."""
    x = dataset.data.reshape(len(dataset.data), 1, 28, 28)/255
    y = dataset.targets
    return x, y


class ConvNet(torch.nn.Module):
    """Toy convolutional neural net."""
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 2, 3, padding=1)
        self.maxpool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(2, 4, 3, padding=1)
        self.dense1 = torch.nn.Linear(4*14*14, 32)
        self.dense2 = torch.nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = x.view(-1, 4*14*14)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x


def train(model, x, y, lr, weight_decay):
    """Train a single epoch."""
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def predict(model, x):
    """Get model prediction scores."""
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
    return y_pred


def get_metrics(y, y_pred):
    """Get loss and accuracy metrics."""
    metrics = {}
    criterion = torch.nn.CrossEntropyLoss()
    metrics["loss"] = criterion(y_pred, y).item()
    _, y_pred_label = torch.max(y_pred, 1)
    metrics["acc"] = (y_pred_label == y).sum().item()/len(y)
    return metrics


def evaluate(model, x, y, metrics_path):
    """Evaluate model and save metrics."""
    scores = predict(model, x)
    metrics = get_metrics(y, scores)
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f)


@click.command()
@click.option("--model_path", required=True)
@click.option("--metrics_path", required=True)
@click.option("--epochs", default=10)
@click.option("--checkpoint", default=0)
def main(model_path, metrics_path, epochs, checkpoint):
    """Train model and evaluate on test data."""
    model = ConvNet()
    # Set output destinations
    # Load model.
    if checkpoint and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    # Load params.
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    # Load train and test data.
    mnist_train = torchvision.datasets.MNIST("data")
    x_train, y_train = transform(mnist_train)
    mnist_test = torchvision.datasets.MNIST("data", train=False)
    x_test, y_test = transform(mnist_test)
    # Iterate over training epochs.
    for i in range(1, epochs+1):
        train(model, x_train, y_train, params["lr"], params["weight_decay"])
        # Evaluate every checkpoint epochs.
        if checkpoint and (not i % checkpoint):
            torch.save(model.state_dict(), model_path)
            evaluate(model, x_test, y_test, metrics_path)
            make_checkpoint()
    # Evaluate and save if not already done via checkpoints.
    if not checkpoint:
        torch.save(model.state_dict(), model_path)
        evaluate(model, x_test, y_test, metrics_path)


if __name__ == "__main__":
    main()
