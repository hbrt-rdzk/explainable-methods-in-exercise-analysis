import argparse
import os

import torch

from src.deep_models.vae_lstm import LSTMVariationalAutoEncoder
from src.trainer import VariationalAutoEncoderTrainer
from src.utils.data import get_data

NUM_JOINTS = 15
SEQUENCE_LENGTH = 25

LATENT_SIZE = 256
NUM_LAYERS = 2
HIDDEN_SIZE = 256

generator = torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VariationalAutoEncoder")
    parser.add_argument(
        "--exercise",
        type=str,
        choices=["squat", "plank", "lunges"],
        default="squat",
        help="Exercise to train the model on",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["LSTM", "CNN"],
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
        args.dataset_dir, args.representation, args.exercise, args.batch_size
    )
    match args.model.lower():
        case "lstm":
            model = LSTMVariationalAutoEncoder(
                SEQUENCE_LENGTH, NUM_JOINTS * 3, HIDDEN_SIZE, LATENT_SIZE, NUM_LAYERS
            )
        case _:
            raise ValueError(f"Model {args.model} not supported")

    loss_fn = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    trainer = VariationalAutoEncoderTrainer(
        model,
        train_dl,
        val_dl,
        loss_fn,
        optimizer,
        device,
    )
    trainer.train(
        args.num_epochs,
        os.path.join(
            args.weights_dir, f"{args.exercise}_{args.representation}_{args.model}.pt"
        ),
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
