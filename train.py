"""Model training and evaluation."""
import os
import click
import torch
import torch.nn.functional as F
import torchvision
import yaml
from dvc.api import make_checkpoint


EPOCHS = 10
CHECKPOINT = 5


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


def evaluate(model, x, y):
    """Evaluate model and save metrics."""
    scores = predict(model, x)
    metrics = get_metrics(y, scores)
    with open("metrics.yaml", "w") as f:
        yaml.dump(metrics, f)


@click.command()
@click.option("--checkpoints/--no-checkpoints", default=False)
def main(checkpoints):
    """Train model and evaluate on test data."""
    model = ConvNet()
    # Load model.
    if checkpoints and os.path.exists("model.pt"):
        model.load_state_dict(torch.load("model.pt"))
    # Load params.
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    # Load train and test data.
    mnist_train = torchvision.datasets.MNIST("data")
    x_train, y_train = transform(mnist_train)
    mnist_test = torchvision.datasets.MNIST("data", train=False)
    x_test, y_test = transform(mnist_test)
    # Iterate over training epochs.
    for i in range(1, EPOCHS+1):
        train(model, x_train, y_train, params["lr"], params["weight_decay"])
        # Evaluate and checkpoint every CHECKPOINT epochs.
        if checkpoints and (not i % CHECKPOINT):
            torch.save(model.state_dict(), "model.pt")
            evaluate(model, x_test, y_test)
            make_checkpoint()
    # Evaluate and save if not already done via checkpoints.
    if not checkpoints:
        torch.save(model.state_dict(), "model.pt")
        evaluate(model, x_test, y_test)


if __name__ == "__main__":
    main()
