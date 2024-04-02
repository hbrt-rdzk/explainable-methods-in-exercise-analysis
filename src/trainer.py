import logging

import torch
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

    def train(self, num_epochs: int = 10) -> list[dict[str, float]]:
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

            results = self.validate()
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {results['train_loss']:.4f}, Train Acc: {results['train_acc']:.4f}, "
                f"Val Loss: {results['val_loss']:.4f}, Val Acc: {results['val_acc']:.4f}"
            )

            learning_results.append(results)
        return learning_results

    def validate(self) -> tuple[torch.Tensor, torch.Tensor]:
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

    @staticmethod
    def __count_correct(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        preds = torch.argmax(y_pred, dim=1)
        return ((preds == y_true).float()).sum()

        x_packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(x_packed)
        out = pad_packed_sequence(out, batch_first=True)
        out = self.fc(out[-1, :, :])
        out = self.softmax(out)
        return out
        # x_packed = pack_padded_sequence(
        #     x, lengths, batch_first=True, enforce_sorted=False
        # )
        # out, _ = self.lstm(x_packed)
        # out = pad_packed_sequence(out, batch_first=True)
        # print(out.shape)

        # out = self.fc(out[-1, :, :])
        # out = self.softmax(out)
        # return out
