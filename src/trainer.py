import logging

import torch
from torch import nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
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

    def train_classifier(
        self, num_epochs: int = 100, weights_path: str = "models/model.pt"
    ) -> list[dict[str, float]]:
        early_stopper = EarlyStopper(patience=50)
        learning_results = []
        self.model.to(self.device)
        for epoch in range(num_epochs):
            self.model.train()
            for inputs, labels, original_lengths in tqdm(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs, original_lengths)
                loss = self.loss_fn(outputs, labels)
                loss.backward()

                self.optimizer.step()

            results = self.validate_classifier()
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {results['train_loss']:.4f}, Train Acc: {results['train_acc']:.4f}, "
                f"Val Loss: {results['val_loss']:.4f}, Val Acc: {results['val_acc']:.4f}"
            )
            learning_results.append(results)

            if early_stopper(results["train_loss"], self.model):
                logger.warning("Eary stopping the training!")
                break

        early_stopper.save_model(weights_path)
        logger.info(f"Best model saved in {weights_path}")
        return learning_results

    def validate_classifier(self) -> tuple[torch.Tensor, torch.Tensor]:
        results = {}
        self.model.eval()
        with torch.no_grad():
            results = {}
            for dataloader in [self.train_loader, self.val_loader]:
                loss = 0
                correct = 0
                total = 0

                for batch, labels, lengths in dataloader:
                    preds = self.model(batch, lengths)
                    loss += self.loss_fn(preds, labels).sum()
                    correct += self.__count_correct(preds, labels)
                    total += labels.size(0)

                if dataloader is self.train_loader:
                    results["train_loss"] = loss / total
                    results["train_acc"] = correct / total
                else:
                    results["val_loss"] = loss / total
                    results["val_acc"] = correct / total

        return results

    def train_autoencoder(
        self, num_epochs: int = 100, weights_path: str = "models/model.pt"
    ) -> list[dict[str, float]]:
        early_stopper = EarlyStopper(patience=50)
        learning_results = []
        self.model.to(self.device)
        for epoch in range(num_epochs):
            self.model.train()
            losses = 0
            for inputs, _, _ in tqdm(self.train_loader):
                inputs = inputs.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, inputs)
                loss.backward()
                losses += loss.item()

                self.optimizer.step()
            if epoch % 10 == 0:
                print(outputs[0][:10], inputs[0][:10])
            logger.info(f"Epoch {epoch+1}/{num_epochs}: " f"Train Loss: {losses}")
            learning_results.append({"train_loss": losses})

            if early_stopper(losses, self.model):
                logger.warning("Eary stopping the training!")
                break

        early_stopper.save_model(weights_path)
        logger.info(f"Best model saved in {weights_path}")
        return learning_results

    @staticmethod
    def __count_correct(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        preds = torch.argmax(y_pred, dim=1)
        return ((preds == y_true).float()).sum()


class EarlyStopper:
    def __init__(self, patience: int = 10, delta: float = 0.0005) -> None:
        self.best_score = None
        self.best_model = None
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_model = None

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
        torch.save(self.best_model.state_dict(), path)
