"""Neural network model with weight freezing capability."""

import torch
from torch import nn


class EEGMotorImageryNet(nn.Module):
    """CNN-based model for motor imagery classification.
    Supports weight freezing for fine-tuning workflow.
    Optimized for 0.5-second windows (125 samples at 250Hz).
    """

    def __init__(
        self, n_channels: int = 64, n_classes: int = 4, n_samples: int = 125
    ):  # 0.5s * 250Hz = 125 samples
        super().__init__()

        # Temporal convolution
        self.conv1 = nn.Conv1d(n_channels, 40, kernel_size=25, padding=12)
        self.bn1 = nn.BatchNorm1d(40)

        # Spatial convolution
        self.conv2 = nn.Conv1d(40, 40, kernel_size=n_channels, groups=40)
        self.bn2 = nn.BatchNorm1d(40)

        # Depthwise separable convolution
        self.conv3 = nn.Conv1d(40, 40, kernel_size=13, padding=6, groups=40)
        self.conv4 = nn.Conv1d(40, 40, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(40)

        # Pooling - adjusted for 125 samples
        # After conv layers: 125 -> pool with kernel 25, stride 5 -> ~20 samples
        self.pool = nn.AvgPool1d(kernel_size=25, stride=5)

        # Classifier - input size depends on pooled output
        # 40 channels * ~20 time samples = ~800 features
        # Using adaptive pooling for robustness
        self.adaptive_pool = nn.AdaptiveAvgPool1d(10)
        self.fc = nn.Linear(40 * 10, n_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Input: (batch, channels, time)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv4(self.conv3(x))))
        x = self.pool(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def freeze_backbone(self):
        """Freeze all layers except classifier."""
        for param in [
            self.conv1,
            self.bn1,
            self.conv2,
            self.bn2,
            self.conv3,
            self.conv4,
            self.bn3,
            self.pool,
            self.adaptive_pool,
        ]:
            for p in param.parameters():
                p.requires_grad = False

        # Keep classifier trainable
        for p in self.fc.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int = 50,
    device: str = "cpu",
    freeze_after: int | None = None,
):
    """Training loop with optional weight freezing.

    Args:
        model: EEGMotorImageryNet model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        device: 'cpu' or 'cuda'
        freeze_after: Epoch number to freeze weights (None = never freeze)

    Returns:
        Trained model and best validation accuracy
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Freeze weights if specified
        if freeze_after is not None and epoch == freeze_after:
            print(f"\nFreezing backbone weights at epoch {epoch}")
            model.freeze_backbone()
            # Update optimizer to only optimize unfrozen parameters
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0.0001,  # Lower LR for fine-tuning
            )

        # Training phase
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_acc = val_correct / val_total
        best_val_acc = max(best_val_acc, val_acc)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

    return model, best_val_acc


def tune_hyperparameters(model, train_loader, val_loader, device="cpu"):
    """Tune hyperparameters on validation set after weights are frozen.

    Args:
        model: Model with frozen backbone
        train_loader: Training data loader
        val_loader: Validation data loader
        device: 'cpu' or 'cuda'

    Returns:
        Best validation accuracy and best hyperparameters dict
    """
    from torch import nn

    # Hyperparameter search space
    learning_rates = [0.0001, 0.0005, 0.001, 0.005]
    weight_decays = [0.0, 1e-5, 1e-4, 1e-3]

    # Save initial model state before hyperparameter tuning
    initial_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    best_val_acc = 0.0
    best_hyperparams = {"lr": 0.0001, "weight_decay": 0.0}

    print("  Testing hyperparameter combinations on validation set...")
    total_combinations = len(learning_rates) * len(weight_decays)
    current_combo = 0

    for lr in learning_rates:
        for wd in weight_decays:
            current_combo += 1

            # Restore model to initial state for fair comparison
            model.load_state_dict({k: v.to(device) for k, v in initial_state.items()})

            # Create fresh optimizer with these hyperparameters
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                weight_decay=wd,
            )
            criterion = nn.CrossEntropyLoss()

            # Quick training: just a few epochs to evaluate hyperparameters
            model.train()
            for _ in range(3):  # 3 epochs for quick evaluation
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # Evaluate on validation set
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_x)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

            val_acc = val_correct / val_total

            print(
                f"    [{current_combo}/{total_combinations}] LR={lr:.4f}, WD={wd:.0e}, Val Acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_hyperparams = {"lr": lr, "weight_decay": wd}
                # Save best model state
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best model state
    if "best_state" in locals():
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    else:
        # Fallback to initial state if no improvement
        model.load_state_dict({k: v.to(device) for k, v in initial_state.items()})

    print(
        f"  Best validation accuracy during tuning: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)"
    )

    return best_val_acc, best_hyperparams


def train_model_with_hyperparams(
    model,
    train_loader,
    val_loader,
    epochs=20,
    device="cpu",
    lr=0.0001,
    weight_decay=0.0,
):
    """Train model with specific hyperparameters (for fine-tuning after hyperparameter tuning).

    Args:
        model: Model (should have frozen backbone)
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        device: 'cpu' or 'cuda'
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)

    Returns:
        Trained model and best validation accuracy
    """
    from torch import nn

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    model.to(device)
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_acc = val_correct / val_total
        best_val_acc = max(best_val_acc, val_acc)

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

    return model, best_val_acc
