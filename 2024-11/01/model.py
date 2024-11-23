import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceWeightedOptimizer:
    """Optimizer wrapper that scales learning rate by node's confidence"""

    def __init__(self, base_optimizer, node):
        self.base_optimizer = base_optimizer
        self.node = node

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self):
        # Get current node confidence
        # We'll use mean confidence over sequence and batch
        current_confidence = self.node.current_confidence.mean()

        # Scale gradients by current confidence
        for param in self.node.parameters():
            if param.grad is not None:
                param.grad *= current_confidence

        self.base_optimizer.step()


class PredictiveNode(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, level):
        super().__init__()
        self.level = level
        self.embedding_dim = embedding_dim

        # Register buffer for current confidence (used by optimizer)
        self.register_buffer("current_confidence", torch.zeros(1))

        # Encoder processes mixed reality input
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=4, dim_feedforward=hidden_dim, batch_first=True
            ),
            num_layers=2,
        )

        # Embedding generator (includes boundary dimension)
        self.embedding_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.embedding_dim),
        )

        # Predictor for next timestep
        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.embedding_dim),
        )

        # Initial NOOP embedding
        self.noop_embedding = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        self.noop_embedding.data[:, :, -1] = 0.0  # boundary flag = 0

    def compute_kl_divergence(self, pred, actual):
        """Compute KL divergence between predicted and actual embeddings"""
        # Normalize embeddings to treat as probability distributions
        pred_norm = F.softmax(pred[:, :-1], dim=-1)  # exclude boundary flag
        actual_norm = F.softmax(actual[:, :-1], dim=-1)  # exclude boundary flag

        kl_div = F.kl_div(pred_norm.log(), actual_norm, reduction="none").sum(dim=-1)

        # Include boundary prediction error
        boundary_error = F.mse_loss(pred[:, -1], actual[:, -1], reduction="none")

        return kl_div + boundary_error

    def compute_confidence(self, kl_div):
        """Convert KL divergence to confidence score"""
        confidence = torch.exp(-kl_div)
        return confidence

    def forward(self, x, prev_prediction=None, prev_pred_confidence=None):
        batch_size, seq_len, _ = x.shape

        if prev_prediction is None:
            prev_prediction = torch.zeros(
                batch_size, seq_len, self.embedding_dim, device=x.device
            )
            prev_pred_confidence = torch.zeros(batch_size, seq_len, device=x.device)

        # Process sequence
        output_sequence = self.noop_embedding.expand(batch_size, seq_len, -1)
        output_confidence = torch.zeros(batch_size, seq_len, device=x.device)

        for t in range(seq_len):
            current_input = x[:, t : t + 1]
            current_pred = prev_prediction[:, t : t + 1]
            current_pred_conf = prev_pred_confidence[:, t : t + 1]

            # Use higher confidence for raw input at level 0
            input_confidence = 1.0 if self.level == 0 else current_pred_conf

            # Mix reality based on confidences
            total_conf = input_confidence + current_pred_conf
            mixed_input = current_input * (
                input_confidence / total_conf
            ) + current_pred * (current_pred_conf / total_conf)

            # Generate embedding
            encoded = self.encoder(mixed_input)
            current_emb = self.embedding_generator(encoded)

            # Compute confidence based on prediction error
            if t > 0:
                kl_div = self.compute_kl_divergence(current_pred, current_emb)
                current_conf = self.compute_confidence(kl_div)
            else:
                current_conf = torch.ones_like(current_pred_conf)

            # Store outputs
            output_sequence[:, t] = current_emb.squeeze(1)
            output_confidence[:, t] = current_conf.squeeze(1)

            # Make prediction for next timestep
            if t < seq_len - 1:
                next_pred = self.predictor(current_emb)
                next_pred_conf = current_conf

        # Update current confidence (used by optimizer)
        self.current_confidence = output_confidence.detach()

        return {
            "embeddings": output_sequence,
            "confidence": output_confidence,
            "next_prediction": next_pred,
            "next_pred_confidence": next_pred_conf,
        }


class PredictiveHierarchicalLM(nn.Module):
    def __init__(
        self, num_levels, input_dim, embedding_dims, hidden_dims, learning_rate
    ):
        super().__init__()
        self.num_levels = num_levels

        # Create hierarchy of predictive nodes
        self.nodes = nn.ModuleList(
            [
                PredictiveNode(
                    input_dim if i == 0 else embedding_dims[i - 1],
                    embedding_dims[i],
                    hidden_dims[i],
                    level=i,
                )
                for i in range(num_levels)
            ]
        )

        # Create confidence-weighted optimizer for each node
        self.optimizers = [
            ConfidenceWeightedOptimizer(
                torch.optim.Adam(node.parameters(), lr=learning_rate), node
            )
            for node in self.nodes
        ]

    def forward(self, x, prev_predictions=None, prev_confidences=None):
        if prev_predictions is None:
            prev_predictions = [None] * self.num_levels
            prev_confidences = [None] * self.num_levels

        level_outputs = []
        current_input = x

        for i, node in enumerate(self.nodes):
            output = node(current_input, prev_predictions[i], prev_confidences[i])
            level_outputs.append(output)
            current_input = output["embeddings"]

        return level_outputs

    def training_step(self, x):
        # Zero all gradients
        for opt in self.optimizers:
            opt.zero_grad()

        # Forward pass
        outputs = self(x)

        # Compute loss
        loss = 0
        for level_output in outputs:
            # Prediction loss
            pred_loss = F.mse_loss(
                level_output["next_prediction"], level_output["embeddings"][:, 1:]
            )
            loss += pred_loss

        # Backward pass
        loss.backward()

        # Update with confidence-weighted gradients
        for opt in self.optimizers:
            opt.step()

        return {
            "loss": loss.item(),
            "confidences": [output["confidence"].mean().item() for output in outputs],
        }


# Training monitoring
class ConfidenceTracker:
    def __init__(self, model):
        self.model = model
        self.confidence_history = [[] for _ in range(model.num_levels)]

    def update(self, outputs):
        for level, output in enumerate(outputs):
            self.confidence_history[level].append(output["confidence"].mean().item())

    def plot_confidence_evolution(self):
        # Could implement visualization of confidence over training
        # Showing how different levels gain confidence at different times
        pass
