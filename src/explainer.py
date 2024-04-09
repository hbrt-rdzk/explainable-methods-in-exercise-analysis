import numpy as np
import torch
import tsaug as tsa
from scipy.interpolate import CubicSpline
from torch import nn


class Explainer:
    def __init__(
        self, model: nn.Module, changable_parameters: list[str], correct_label: int
    ) -> None:
        self.model = model
        self.changable_parameters = changable_parameters
        self.correct_label = correct_label

    def explain(self, sample: torch.Tensor, sample_length: int) -> torch.Tensor:
        start_sample_probabilities = self.model(sample, [sample_length])
        best_sample = sample
        best_score = start_sample_probabilities.squeeze()[0]
        print("start score", best_score)

        sigmas = np.linspace(0.01, 0.05, 10)

        for sigma in [*sigmas] * 100:
            augmented_time_serieses = []
            for parameter in range(sample.shape[2]):
                new_sample = sample[0, :sample_length, parameter]
                if parameter in self.changable_parameters:
                    modified_timeseries = self.modify_timeseries(
                        new_sample, sigma, knot=4
                    )
                    augmented_time_serieses.append(modified_timeseries)
                else:
                    augmented_time_serieses.append(new_sample)

            modified_sample = torch.stack(augmented_time_serieses, dim=1)
            new_sample_probabilities = self.model(
                modified_sample.unsqueeze(0), [sample_length]
            )
            if new_sample_probabilities.squeeze()[0] > best_score:
                best_score = new_sample_probabilities.squeeze()[0]
                best_sample = modified_sample
                print("new best score", best_score)

        return best_sample

    def modify_timeseries(
        self, ts: torch.Tensor, sigma: float = 0.2, knot: int = 4
    ) -> torch.Tensor:
        ts_ = ts.clone()

        random_knot_magnitudes = np.random.normal(1, sigma, size=knot)

        knot_points = np.linspace(0, len(ts_) - 1, knot).astype(int)
        spline = CubicSpline(knot_points, random_knot_magnitudes)

        warp_factors = spline(np.arange(len(ts_)))
        warped_ts = ts_ * warp_factors
        warped_ts = torch.clip(warped_ts, min=0, max=180)
        return warped_ts.float()
