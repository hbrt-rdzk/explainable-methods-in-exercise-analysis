import argparse

import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier

from src.trainer import ClassifierTrainer
from src.utils.data import encode_samples_to_latent, get_data
from src.vae_architectures.lstm import LSTMVariationalAutoEncoder

NUM_JOINTS = 15
SEQUENCE_LENGTH = 25
HIDDEN_SIZE = 256
LATENT_SIZE = 256
NUM_LAYERS = 2


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
        default="squat",
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
    train_dl, val_dl = get_data(args.dataset_dir, args.representation, args.exercise)
    train_data = torch.stack(train_dl.dataset.data)
    val_data = torch.stack(val_dl.dataset.data)

    architecture_name = args.autoencoder.split(".")[0].split("/")[-1].split("_")[-1]
    match architecture_name.lower():
        case "lstm":
            model = LSTMVariationalAutoEncoder(
                SEQUENCE_LENGTH, NUM_JOINTS * 3, HIDDEN_SIZE, LATENT_SIZE, NUM_LAYERS
            )
        case _:
            raise ValueError("Model name not supported")

    model.load_state_dict(torch.load(args.autoencoder))

    latent_train_data = encode_samples_to_latent(model, train_data).detach().numpy()
    latent_test_data = encode_samples_to_latent(model, val_data).detach().numpy()

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
    trainer.save(args.weights_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
