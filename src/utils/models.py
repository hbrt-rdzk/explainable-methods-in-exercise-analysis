import pickle

import torch
from sklearn.base import BaseEstimator

from src.utils.constants import (HIDDEN_SIZE, LATENT_SIZE, NUM_JOINTS,
                                 NUM_LAYERS, SEQUENCE_LENGTH)
from src.vae_architectures.graph_cnn import GraphVariationalAutoEncoder
from src.vae_architectures.lstm import LSTMVariationalAutoEncoder
from src.vae_architectures.signal_cnn import SignalCNNVariationalAutoEncoder


def load_models(vae_architecture_name: str, vae_weights_path: str, classifier_params_path: str) -> list[torch.Module, BaseEstimator]:
    vae = load_vae(vae_architecture_name, vae_weights_path)
    clf = load_clf(classifier_params_path)
    return vae, clf

def load_vae(vae_architecture_name: str, vae_weights_path: str) -> torch.Module:
    match vae_architecture_name.lower():
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
            raise ValueError("Model name not supported")

    vae.load_state_dict(torch.load(vae_weights_path))
    return vae

def load_clf(classifier_params_path: str) -> BaseEstimator:
    with open(classifier_params_path, "rb") as f:
        clf = pickle.load(f)
    return clf
