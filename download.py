"""Download training and test data."""
import click
import torchvision


@click.command()
def download():
    torchvision.datasets.MNIST("data", download=True)


if __name__ == "__main__":
    download()
