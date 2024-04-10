import argparse
import pickle
import warnings

import dice_ml
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator

from src.ae_architectures.lstm import LSTMAutoEncoder
from src.utils.data import generate_latent_samples, get_data

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
    return parser.parse_args()


def explain(clf: BaseEstimator, data: np.ndarray, labels: list) -> torch.Tensor:
    train_df = pd.DataFrame(data)
    train_df = train_df.rename(str, axis="columns")
    features = list(train_df.columns)

    train_df["label"] = labels
    m = dice_ml.Model(model=clf, backend="sklearn")
    d = dice_ml.Data(
        dataframe=train_df, continuous_features=features, outcome_name="label"
    )
    exp = dice_ml.Dice(d, m)

    query_instance = train_df.iloc[[1], :-1]
    explanation = exp.generate_counterfactuals(
        query_instance, total_CFs=1, desired_class="opposite"
    )
    return torch.tensor(
        explanation.cf_examples_list[0].final_cfs_df.values[:, :-1]
    ).float()


def main(args: argparse.Namespace) -> None:
    train_dl, val_dl = get_data(
        args.dataset_dir, args.representation, args.exercise, args.batch_size
    )
    architecture_name = args.autoencoder.split(".")[0].split("/")[-1].split("_")[-1]
    match architecture_name.lower():
        case "lstm":
            ae = LSTMAutoEncoder(
                SEQUENCE_LENGTH, NUM_JOINTS * 3, HIDDEN_SIZE, LATENT_SIZE, NUM_LAYERS
            )
        case _:
            raise ValueError("Model name not supported")

    ae.load_state_dict(torch.load(args.autoencoder))

    latent_train_data = generate_latent_samples(ae, train_dl)
    latent_test_data = generate_latent_samples(ae, val_dl)

    ae.load_state_dict(torch.load(args.autoencoder))
    with open(args.classifier, "rb") as f:
        clf = pickle.load(f)

    binary_train_labels = np.array(
        [1 if label == 0 else 0 for label in train_dl.dataset.labels_encoded]
    )
    fixed_sample = explain(clf, latent_train_data, binary_train_labels)
    dct_sample = ae.decoder(fixed_sample)


if __name__ == "__main__":
    args = parse_args()
    main(args)
