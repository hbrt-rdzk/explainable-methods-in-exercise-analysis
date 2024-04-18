import argparse
import pickle
import warnings

import torch
import os
from src.explainer import Explainer
from src.utils.data import decode_dct, get_data
from src.vae_architectures.lstm import LSTMVariationalAutoEncoder
from utils.visualization import get_3D_animation_comparison

warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*X has feature names.*"
)

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
        "--classifier",
        "-clf",
        type=str,
        required=True,
        default="models",
        help="Path to the trained classifier model",
    )
    parser.add_argument(
        "--exercise",
        type=str,
        choices=["squat", "plank", "lunges"],
        default="squat",
        help="Exercise to train the model on",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=1,
        help="Sample number to explain",
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
        "--output_dir",
        type=str,
        default="docs/cfe_videos",
        help="Output for comaprison video",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    train_dl, val_dl = get_data(args.dataset_dir, args.representation, args.exercise)
    architecture_name = args.autoencoder.split(".")[0].split("/")[-1].split("_")[-1]
    match architecture_name.lower():
        case "lstm":
            ae = LSTMVariationalAutoEncoder(
                SEQUENCE_LENGTH, NUM_JOINTS * 3, HIDDEN_SIZE, LATENT_SIZE, NUM_LAYERS
            )
        case _:
            raise ValueError("Model name not supported")

    ae.load_state_dict(torch.load(args.autoencoder))
    with open(args.classifier, "rb") as f:
        clf = pickle.load(f)

    query_dct_data = val_dl.dataset.data[args.sample_num]
    query_dct_length = val_dl.dataset.lengths[args.sample_num]

    explainer = Explainer(ae, clf, train_dl)
    fixed_dct_sample = explainer.generate_cf(query_dct_data.detach().numpy())

    query_sample = decode_dct(query_dct_data, query_dct_length)
    fixed_sample = decode_dct(fixed_dct_sample, query_dct_length)

    anim = get_3D_animation_comparison(query_sample, fixed_sample)
    anim.save(
        os.path.join(args.output_dir, f"{args.sample_num}_comparison.mp4"),
        writer="ffmpeg",
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
