from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ConfidenceTracker, PredictiveHierarchicalLM
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset


class ShakespeareDataset(Dataset):
    def __init__(self, sequence_length=128, split="train"):
        super().__init__()

        # Download Shakespeare text if not exists
        data_path = Path("shakespeare.txt")
        if not data_path.exists():
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            text = requests.get(url).text
            data_path.write_text(text)

        text = data_path.read_text()

        # Create vocabulary
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        # Convert text to indices
        data = torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)

        # Split data
        n = int(0.9 * len(data))
        self.data = data[:n] if split == "train" else data[n:]
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length - 1

    def __getitem__(self, idx):
        # Get sequence and target
        x = self.data[idx : idx + self.sequence_length]
        y = self.data[idx + 1 : idx + self.sequence_length + 1]

        # Convert to one-hot encodings
        x_onehot = F.one_hot(x, num_classes=self.vocab_size).float()

        return x_onehot, y


class PHLMLightning(pl.LightningModule):

    def __init__(
        self,
        vocab_size,
        num_levels=3,
        embedding_dims=[128, 256, 512],
        hidden_dims=[256, 512, 1024],
        learning_rate=1e-4,
        sequence_length=128,
        plot_interval=100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.plot_interval = plot_interval

        # Create model
        self.model = PredictiveHierarchicalLM(
            num_levels=num_levels,
            input_dim=vocab_size,
            embedding_dims=embedding_dims,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
        )

        # Initialize confidence tracker
        self.confidence_tracker = ConfidenceTracker(self.model)

        # For decoding predictions
        self.vocab_size = vocab_size

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)

        # Track confidences
        self.confidence_tracker.update(outputs)

        # Compute loss for each level
        total_loss = 0
        level_losses = []

        for level, output in enumerate(outputs):
            # Prediction loss for embeddings
            pred_loss = F.mse_loss(
                output["next_prediction"], output["embeddings"][:, 1:]
            )

            # Add to total loss
            level_losses.append(pred_loss)
            total_loss += pred_loss

            # Log level-specific metrics
            self.log(f"train_loss_level_{level}", pred_loss)
            self.log(f"train_confidence_level_{level}", output["confidence"].mean())

        # Log total loss
        self.log("train_loss", total_loss)

        # Add visualization calls
        if batch_idx % self.plot_interval == 0:
            self.confidence_tracker.plot_confidence_evolution()
            self.confidence_tracker.plot_confidence_distribution()
            self.confidence_tracker.plot_confidence_correlation()

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)

        # Similar to training step but for validation
        total_loss = 0
        for level, output in enumerate(outputs):
            pred_loss = F.mse_loss(
                output["next_prediction"], output["embeddings"][:, 1:]
            )
            total_loss += pred_loss
            self.log(f"val_loss_level_{level}", pred_loss)
            self.log(f"val_confidence_level_{level}", output["confidence"].mean())

        self.log("val_loss", total_loss)

        return total_loss

    def configure_optimizers(self):
        # Each node has its own confidence-weighted optimizer
        # Return them all for Lightning to manage
        # return self.model.optimizers.copy()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def generate(self, seed_text, max_length=100):
        """Generate text using the trained model"""
        self.eval()
        with torch.no_grad():
            # Convert seed text to input tensor
            chars = [self.char_to_idx[ch] for ch in seed_text]
            x = torch.tensor(chars, device=self.device)
            x = F.one_hot(x, num_classes=self.vocab_size).float().unsqueeze(0)

            generated = list(seed_text)

            for _ in range(max_length):
                # Get model predictions
                outputs = self(x)

                # Use lowest level predictions for character generation
                level0_output = outputs[0]
                next_char_logits = level0_output["next_prediction"][
                    0, -1, :-1
                ]  # Exclude boundary dim

                # Sample from the predictions
                probs = F.softmax(next_char_logits, dim=-1)
                next_char_idx = torch.multinomial(probs, 1).item()

                # Append to generated text
                generated.append(self.idx_to_char[next_char_idx])

                # Update input for next iteration
                next_char_tensor = (
                    F.one_hot(
                        torch.tensor([next_char_idx]), num_classes=self.vocab_size
                    )
                    .float()
                    .unsqueeze(0)
                )
                x = torch.cat([x[:, 1:], next_char_tensor], dim=1)

        return "".join(generated)


def train_phlm():
    # Set up data
    train_dataset = ShakespeareDataset(split="train")
    val_dataset = ShakespeareDataset(split="val")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Create model
    model = PHLMLightning(
        vocab_size=72,  # train_dataset.vocab_size,
        num_levels=3,
        embedding_dims=[128, 256, 512],
        hidden_dims=[256, 512, 1024],
        learning_rate=1e-4,
    )

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="phlm-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    return model, trainer


# Example usage:
if __name__ == "__main__":
    # Train model
    model, trainer = train_phlm()

    # Generate some text
    seed_text = "To be or not to be"
    generated_text = model.generate(seed_text)
    print(f"Generated text:\n{generated_text}")

    # Plot confidence evolution
    model.confidence_tracker.plot_confidence_evolution()
