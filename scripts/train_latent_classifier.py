import argparse
import os

import torch
from sklearn.tree import DecisionTreeClassifier

from src.trainer import ClassifierTrainer
from src.utils.data import encode_samples_to_latent, get_data
from src.utils.models import load_vae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classifier")
    parser.add_argument(
        "--autoencoder",
        "-ae",
        type=str,
        required=True,
        help="Path to the trained autoencoder model (.pt file)",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="models",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--exercise",
        type=str,
        choices=["squat", "plank", "lunges"],
        help="Exercise to train the model on",
    )
    parser.add_argument(
        "--representation",
        type=str,
        choices=["joints", "angles", "dct"],
        default="dct",
        help="Representation to use for training",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data",
        help="Directory where the dataset is stored",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    train_dl, val_dl = get_data(args.dataset_dir, args.exercise, args.representation)
    train_data = torch.stack(train_dl.dataset.data)
    val_data = torch.stack(val_dl.dataset.data)

    vae_architecture_name = args.autoencoder.split(".")[0].split("/")[-1].split("_")[-1]
    vae = load_vae(vae_architecture_name, args.autoencoder)

    latent_train_data = encode_samples_to_latent(vae, train_data).detach().numpy()
    latent_test_data = encode_samples_to_latent(vae, val_data).detach().numpy()

    train_labels, test_labels = (
        train_dl.dataset.labels_encoded.numpy(),
        val_dl.dataset.labels_encoded.numpy(),
    )
    trainer = ClassifierTrainer(
        DecisionTreeClassifier,
        latent_train_data,
        latent_test_data,
        train_labels,
        test_labels,
    )
    trainer.train()
    trainer.save(args.weights_dir, f"{vae_architecture_name}_")


if __name__ == "__main__":
    args = parse_args()
    main(args)
