import argparse
import os
import pickle
import warnings

import torch

from src.explainer import Explainer
from src.utils.constants import OPENPOSE_ANGLES
from src.utils.data import (decode_dct, get_angles_from_joints, get_data,
                            get_random_sample)
from src.utils.evaluation import get_dtw_score
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
        help="Exercise to train the model on",
    )
    parser.add_argument(
        "--sample_label",
        type=str,
        default=1,
        help="Sample label to explain",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["cf", "closest"],
        default="cf",
        help="Generation of the new sample method",
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
    train_dl, val_dl = get_data(args.dataset_dir, args.exercise, args.representation)
    query_sample_dct, query_sample_length = get_random_sample(val_dl, args.sample_label)
    correct_sample_dct, correct_sample_length = get_random_sample(
        val_dl, desired_label="correct"
    )

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

    explainer = Explainer(ae, clf, train_dl, args.exercise)
    match args.method:
        case "cf":
            fixed_sample_dct = explainer.generate_cf(query_sample_dct.detach().numpy())
        case "closest":
            fixed_sample_dct = explainer.get_closest_correct(query_sample_dct.detach().numpy())
        case _:
            raise ValueError("Method not supported")

    correct_sample = decode_dct(correct_sample_dct, correct_sample_length)
    query_sample = decode_dct(query_sample_dct, query_sample_length)
    fixed_query_sample = decode_dct(fixed_sample_dct, query_sample_length)

    correct_sample_angles = get_angles_from_joints(
        correct_sample.reshape(-1, 15, 3), OPENPOSE_ANGLES
    )
    query_sample_angles = get_angles_from_joints(
        query_sample.reshape(-1, 15, 3), OPENPOSE_ANGLES
    )
    fixed_query_sample_angles = get_angles_from_joints(
        fixed_query_sample.reshape(-1, 15, 3), OPENPOSE_ANGLES
    )

    print(
        f"Statistical results for incorrect sample:\n{explainer.statistical_classification(query_sample_angles)}"
    )
    print(
        f"Statistical results for fixed sample:\n{explainer.statistical_classification(fixed_query_sample_angles)}"
    )
    print(
        f"DTW score correct - incorrect: {get_dtw_score(correct_sample_angles[explainer.important_angles], query_sample_angles[explainer.important_angles]):.4f}"
    )
    print(
        f"DTW score correct - fixed: {get_dtw_score(correct_sample_angles[explainer.important_angles], fixed_query_sample_angles[explainer.important_angles]):.4f}\n"
    )

    anim = get_3D_animation_comparison(query_sample, fixed_query_sample)
    anim.save(
        os.path.join(args.output_dir, args.exercise, f"{args.sample_label}_fixed.mp4"),
        writer="ffmpeg",
    )
    print(f"Fixed sample comparison saved in {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
