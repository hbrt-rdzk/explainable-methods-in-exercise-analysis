import argparse
import os
import warnings

import torch

from src.explainer import Explainer
from src.utils.constants import OPENPOSE_ANGLES
from src.utils.data import (decode_dct, decode_samples_from_latent,
                            get_angles_from_joints, get_data,
                            get_random_sample)
from src.utils.evaluation import get_dtw_score
from src.utils.models import load_models
from src.utils.visualization import save_anim
from utils.visualization import get_3D_animation, get_3D_animation_comparison

warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*X has feature names.*"
)


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
    vae_architecture_name = args.autoencoder.split(".")[0].split("/")[-1].split("_")[-1]
    vae, clf = load_models(vae_architecture_name, args.autoencoder ,args.classifer)
    
    explainer = Explainer(vae, clf, train_dl, args.exercise, threshold=20)
    match args.method:
        case "cf":
            fixed_sample_latent = explainer.generate_cf(query_sample_dct.detach().numpy())
        case "closest":
            fixed_sample_latent = explainer.get_closest_correct(query_sample_dct.detach().numpy())
        case _:
            raise ValueError("Method not supported")
    fixed_sample_dct = decode_samples_from_latent(vae, torch.Tensor(fixed_sample_latent)).detach().numpy().squeeze()

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

    incorrect_classification_report = explainer.statistical_classification(
        query_sample_angles
    )
    fixed_classification_report = explainer.statistical_classification(
        fixed_query_sample_angles
    )

    incorrect_dtw_score = get_dtw_score(
        correct_sample_angles[explainer.important_angles],
        query_sample_angles[explainer.important_angles],
    )
    fixed_dtw_score = get_dtw_score(
        correct_sample_angles[explainer.important_angles],
        fixed_query_sample_angles[explainer.important_angles],
    )

    print(
        f"""
        Statistical results for incorrect sample:\n{incorrect_classification_report}
        Statistical results for fixed sample:\n{fixed_classification_report}
        DTW score correct - incorrect: {incorrect_dtw_score:.4f}
        DTW score correct - fixed: {fixed_dtw_score:.4f}\n"""
    )
    anim_path = os.path.join(args.output_dir, args.exercise, vae_architecture_name, args.sample_label)

    bad_anim = get_3D_animation(query_sample, color="red", is_plank=True if args.exercise == 'plank' else False)
    fixed_anim = get_3D_animation(fixed_query_sample, color="green", is_plank=True if args.exercise == 'plank' else False)
    comparitson_anim = get_3D_animation_comparison(
        query_sample, fixed_query_sample, args.sample_label, is_plank=True if args.exercise == 'plank' else False
    )

    save_anim(bad_anim, anim_path + "_bad.mp4")
    save_anim(fixed_anim, anim_path + "_fixed.mp4")
    save_anim(comparitson_anim, anim_path + "_comparison.mp4")

    print(f"Animations saved in {anim_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
