import logging
import os
import pickle as pkl

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report
from torch import nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VariationalAutoEncoderTrainer:
    """Trainer of all Variational Autoencoder classes"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device="cpu",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train(
        self, num_epochs: int = 100, weights_path: str = "models/model.pt"
    ) -> list[dict[str, float]]:
        """Train variational Autoencoder"""
        early_stopper = EarlyStopper(patience=50)
        learning_results = []
        self.model.to(self.device)
        for epoch in range(num_epochs):
            self.model.train()
            losses = 0
            mse_loss = 0
            kld_loss = 0

            for inputs, _, _ in tqdm(self.train_loader):
                inputs = inputs.to(self.device)
                self.optimizer.zero_grad()
                outputs, means, log_vars = self.model(inputs)

                base_loss = self.loss_fn(outputs, inputs)
                kld_loss = -0.5 * torch.sum(
                    1 + log_vars - means.pow(2) - log_vars.exp()
                )
                loss = base_loss + kld_loss
                loss.backward()
                self.optimizer.step()

                losses += loss.item()
                mse_loss += base_loss.item()
                kld_loss += kld_loss.item()

            val_mse_loss, val_kld_loss = self.evaluate()
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Val Loss: {val_mse_loss + val_kld_loss}, Train Loss: {losses}"
            )

            learning_results.append(
                {
                    "train_kld_loss": kld_loss,
                    "train_mse_loss": mse_loss,
                    "val_kld_loss": val_kld_loss,
                    "val_mse_loss": val_mse_loss,
                }
            )
            if early_stopper(losses, self.model):
                logger.warning("Eary stopping the training!")
                break

        early_stopper.save_model(weights_path)
        logger.info(f"Best model saved in {weights_path}")
        return learning_results

    def evaluate(self) -> tuple[float, float]:
        """Evaluate Variational Autoencoder on validation dataset"""
        self.model.eval()
        mse_loss = 0
        kld_loss = 0
        with torch.no_grad():
            for inputs, _, _ in self.val_loader:
                inputs = inputs.to(self.device)
                outputs, means, log_vars = self.model(inputs)
                base_loss = self.loss_fn(outputs, inputs)
                kld_loss = -0.5 * torch.sum(
                    1 + log_vars - means.pow(2) - log_vars.exp()
                )
                mse_loss += base_loss.item()
                kld_loss += kld_loss.item()
        return mse_loss, kld_loss


class EarlyStopper:
    """Early stop training if loss is not descending"""

    def __init__(self, patience: int = 10, delta: float = 0.0005) -> None:
        self.best_score = None
        self.best_model = None
        self.patience = patience
        self.delta = delta
        self.counter = 0

    def __call__(self, train_loss: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = train_loss
            self.best_model = model
        elif train_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = train_loss
            self.best_model = model
            self.counter = 0
        return False

    def save_model(self, path):
        """Save best model to dessired path"""
        torch.save(self.best_model.state_dict(), path)


class ClassifierTrainer:
    """Trainer of the Sklearn's classification model"""

    def __init__(
        self,
        estimator: BaseEstimator,
        train_data: np.ndarray,
        test_data: np.ndarray,
        train_labels: list,
        test_labels: list,
    ) -> None:
        self.estimator = estimator()
        self.train_data = train_data
        self.test_data = test_data
        self.train_labels = train_labels
        self.test_labels = test_labels

    def train(self):
        """Train classifier on the latent space"""
        binary_train_labels = [1 if label == 0 else 0 for label in self.train_labels]
        binary_test_labels = [1 if label == 0 else 0 for label in self.test_labels]
        self.estimator.fit(self.train_data, binary_train_labels)
        self.predictions = self.estimator.predict(self.test_data)
        logger.info(
            f"Test data results: {classification_report(binary_test_labels, self.predictions)}"
        )

    def save(self, path: str, file_name: str) -> None:
        """Save classificaiton model"""
        pkl.dump(self.estimator, open(os.path.join(path, file_name + "clf.pkl"), "wb"))
        logger.info(f"Classifier saved in {path}")
