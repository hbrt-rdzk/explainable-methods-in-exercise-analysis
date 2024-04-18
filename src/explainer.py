import dice_ml
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from torch import nn
from torch.utils.data import DataLoader

from src.utils.data import decode_samples_from_latent, encode_samples_to_latent


class Explainer:
    def __init__(
        self, autoencoder: nn.Module, classifier: BaseEstimator, dl: DataLoader
    ) -> None:
        self.autoencoder = autoencoder
        self.classifier = classifier
        self.dl = dl
        self.data = dl.dataset.data
        self.labels = dl.dataset.labels
        self.binary_labels = np.array([1 if label == 0 else 0 for label in self.labels])
        self.latent_data = (
            encode_samples_to_latent(self.autoencoder, self.data).detach().numpy()
        )

    def generate_cf(self, query: np.ndarray) -> np.ndarray:
        train_df = pd.DataFrame(self.latent_data)
        train_df = train_df.rename(str, axis="columns")
        features = list(train_df.columns)

        latent_query = (
            encode_samples_to_latent(self.autoencoder, [torch.Tensor(query)])
            .detach()
            .numpy()
        )
        latent_query_df = pd.DataFrame(latent_query, columns=features)

        train_df["label"] = self.binary_labels
        m = dice_ml.Model(model=self.classifier, backend="sklearn")
        d = dice_ml.Data(
            dataframe=train_df, continuous_features=features, outcome_name="label"
        )
        exp = dice_ml.Dice(d, m)

        explanation = exp.generate_counterfactuals(
            latent_query_df,
            total_CFs=1,
            desired_class="opposite",
            stopping_threshold=0.99,
        )
        cf = explanation.cf_examples_list[0].final_cfs_df.values[:, :-1]

        return (
            decode_samples_from_latent(self.autoencoder, torch.Tensor(cf))
            .detach()
            .numpy()
            .squeeze()
        )
