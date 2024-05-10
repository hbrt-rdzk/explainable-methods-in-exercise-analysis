import argparse
import os

import torch

from src.trainer import VariationalAutoEncoderTrainer
from src.utils.constants import (HIDDEN_SIZE, LATENT_SIZE, NUM_JOINTS,
                                 NUM_LAYERS, SEQUENCE_LENGTH)
from src.utils.data import get_data
from src.vae_architectures.graph_cnn import GraphVariationalAutoEncoder
from src.vae_architectures.lstm import LSTMVariationalAutoEncoder
from src.vae_architectures.signal_cnn import SignalCNNVariationalAutoEncoder

generator = torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VariationalAutoEncoder")
    parser.add_argument(
        "--exercise",
        type=str,
        choices=["squat", "plank", "lunges"],
        help="Exercise to train the model on",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lstm", "cnn", "graph"],
        default="lstm",
        help="Model architecture to train",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="models",
        help="Directory to save the trained model",
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.002,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=350,
        help="Number of epochs to train the model",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    train_dl, val_dl = get_data(
        args.dataset_dir, args.exercise, args.representation, args.batch_size
    )
    match args.model.lower():
        case "lstm":
            vae = LSTMVariationalAutoEncoder(
                SEQUENCE_LENGTH, NUM_JOINTS * 3, HIDDEN_SIZE, LATENT_SIZE, NUM_LAYERS
            )
        case "cnn":
            vae = SignalCNNVariationalAutoEncoder(
                SEQUENCE_LENGTH, NUM_JOINTS * 3, HIDDEN_SIZE, LATENT_SIZE
            )
        case "graph":
            vae = GraphVariationalAutoEncoder(
                SEQUENCE_LENGTH, NUM_JOINTS * 3, HIDDEN_SIZE, LATENT_SIZE
            )
        case _:
            raise ValueError(f"Model {args.model} not supported")

    loss_fn = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(vae.parameters(), args.learning_rate)
    trainer = VariationalAutoEncoderTrainer(
        vae,
        train_dl,
        val_dl,
        loss_fn,
        optimizer,
        device,
    )
    trainer.train(
        args.num_epochs,
        os.path.join(args.weights_dir, f"{args.representation}_{args.model}.pt"),
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
