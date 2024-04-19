import dice_ml
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator
from torch import nn
from torch.utils.data import DataLoader

from src.utils.data import (decode_samples_from_latent,
                            encode_samples_to_latent, segment_signal)


class Explainer:
    """This class implements functionality of explaining user's
    mistakes during exercise performance
    """

    def __init__(
        self,
        autoencoder: nn.Module,
        classifier: BaseEstimator,
        dl: DataLoader,
        exercise: str,
        threshold: float = 10.0,
    ) -> None:
        self.autoencoder = autoencoder
        self.classifier = classifier
        self.dl = dl
        self.exercise = exercise

        self.data = torch.stack(dl.dataset.data)
        self.labels = dl.dataset.labels
        self.binary_labels = np.array(
            [1 if label == "correct" else 0 for label in self.labels]
        )
        self.latent_train_data = (
            encode_samples_to_latent(self.autoencoder, self.data).detach().numpy()
        )

        with open(f"configs/{exercise}.yaml", "r") as file:
            file_data = yaml.safe_load(file)
            self.important_angles = file_data["important_angles"]
            self.reference_table = pd.DataFrame(
                file_data["reference_table"]
            ).transpose()
        self.classification_threshold = threshold

    def generate_cf(self, query: np.ndarray) -> np.ndarray:
        """Generate CounterFactual perputation to the provided query sample"""
        train_df = pd.DataFrame(self.latent_train_data)
        train_df = train_df.rename(str, axis="columns")
        features = list(train_df.columns)

        latent_query = (
            encode_samples_to_latent(self.autoencoder, torch.Tensor(query).unsqueeze(0))
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

    def get_closest_correct(self, query: np.ndarray) -> np.ndarray:
        latent_query = (
            encode_samples_to_latent(self.autoencoder, torch.Tensor(query).unsqueeze(0))
            .detach()
            .numpy()
        )
        closest_correct_instances = cdist(
            latent_query, self.latent_train_data
        ).squeeze()
        mask = np.where(self.binary_labels == 1)[0]

        mask_argmin = closest_correct_instances[mask].argmin()
        cf_id = mask[mask_argmin]
        cf = self.latent_train_data[cf_id]

        if cf.ndim == 1:
            cf = np.expand_dims(cf, 0)

        return (
            decode_samples_from_latent(self.autoencoder, torch.Tensor(cf))
            .detach()
            .numpy()
            .squeeze()
        )

    def statistical_classification(self, query_angles: pd.DataFrame) -> str:
        phases_names = self.reference_table.index.values
        reference_angles = self.reference_table.reset_index(drop=True)
        phases = segment_signal(query_angles, self.important_angles)
        phases = phases.reset_index(drop=True)

        results = reference_angles - phases
        results["phase"] = phases_names
        results = results.set_index("phase")
        result_str = ""
        for phase, result in results.iterrows():
            wrong_angles = result.loc[result.abs() > self.classification_threshold]
            for angle_name, difference in wrong_angles.items():
                result_str += f"At {phase} phase {angle_name} angle was different from reference by {difference} degrees.\n"

        if not result_str:
            result_str = "correct"
        return result_str
