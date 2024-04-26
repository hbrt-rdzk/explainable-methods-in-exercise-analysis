import torch


def reparameterization_trick(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std)
    z = mean + std * epsilon
    return z
