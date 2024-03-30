import torch


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        loss_fn,
        optimizer,
        device="cpu",
    ):
        self.model = model
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train(self, num_epochs=10):
        self.model.to(self.device)
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)
            val_loss = 0.0
            # self.model.eval()
            # with torch.no_grad():
            #     for inputs, labels in self.val_loader:
            #         inputs, labels = inputs.to(self.device), labels.to(self.device)
            #         outputs = self.model(inputs)
            #         loss = self.criterion(outputs, labels)
            #         val_loss += loss.item()
            #     val_loss /= len(self.val_loader)
            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            self.scheduler.step(val_loss)
