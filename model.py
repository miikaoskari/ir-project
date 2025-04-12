import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

os.makedirs("outputs", exist_ok=True)

data = pd.read_csv("data/df_total.csv")

# create additional features
data["gnss_error_x"] = data["gnss_x"] - data["true_x"]
data["gnss_error_y"] = data["gnss_y"] - data["true_y"]
data["gnss_error"] = np.sqrt(data["gnss_error_x"] ** 2 + data["gnss_error_y"] ** 2)

# use time differences directly instead of timestamps
data["time_delta"] = np.gradient(data["timestamps"])


# handle potential division by zero in gradient calculations
def safe_gradient(x, y):
    """Calculate gradient safely, avoiding division by zero"""
    try:
        # replace zero deltas with a small value
        y_diff = np.diff(y)
        x_diff = np.diff(x)
        x_diff = np.where(x_diff == 0, 1e-10, x_diff)  # avoid division by zero
        grad = np.zeros_like(y, dtype=float)
        grad[1:] = y_diff / x_diff
        # Forward fill the first element
        grad[0] = grad[1] if len(grad) > 1 else 0
        return grad
    except Exception as e:
        print(f"Gradient calculation error: {e}")
        return np.zeros_like(y, dtype=float)


data["velocity_x"] = safe_gradient(data["timestamps"], data["true_x"])
data["velocity_y"] = safe_gradient(data["timestamps"], data["true_y"])
data["speed"] = np.sqrt(data["velocity_x"] ** 2 + data["velocity_y"] ** 2)
data["acceleration"] = safe_gradient(data["timestamps"], data["speed"])

# create time-based features without using datetime conversion
# use normalized time values (mod 24 hours)
one_day_seconds = 24 * 60 * 60
normalized_hours = (data["timestamps"] % one_day_seconds) / (60 * 60)
data["hour_sin"] = np.sin(2 * np.pi * normalized_hours / 24)
data["hour_cos"] = np.cos(2 * np.pi * normalized_hours / 24)

# select features for GNSS correction model
X = np.column_stack(
    [
        data["gnss_x"],
        data["gnss_y"],
        data["ax"],
        data["ay"],
        data["time_delta"],
        data["hour_sin"],
        data["hour_cos"],
        data["gnss_error_x"].shift(1).fillna(0),
        data["gnss_error_y"].shift(1).fillna(0),
    ]
)

# target: error between GNSS and true position
y = np.column_stack([data["true_x"] - data["gnss_x"], data["true_y"] - data["gnss_y"]])

# split 80/20 for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
gnss_train = data.iloc[: -int(len(data) * 0.2)][["gnss_x", "gnss_y"]].values
gnss_test = data.iloc[-int(len(data) * 0.2) :][["gnss_x", "gnss_y"]].values
true_test = data.iloc[-int(len(data) * 0.2) :][["true_x", "true_y"]].values

# handle outliers
scaler_X = RobustScaler()
scaler_y = RobustScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# create sequence data for RNN-based processing
seq_length = 10


def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i : i + seq_length])
        y_seq.append(y[i + seq_length - 1])  # predict the last point in the sequence
    return np.array(X_seq), np.array(y_seq)


X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

X_train_seq_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_seq_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
X_test_seq_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
y_test_seq_tensor = torch.tensor(y_test_seq, dtype=torch.float32)

# create TensorDatasets and DataLoaders for more efficient training
train_dataset = TensorDataset(X_train_seq_tensor, y_train_seq_tensor)
test_dataset = TensorDataset(X_test_seq_tensor, y_test_seq_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Input shape: {X_train_seq.shape}, Output shape: {y_train_seq.shape}")


class HybridPositioningModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, lstm_layers=2, dropout=0.2):
        super(HybridPositioningModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # CNN for extracting spatial features
        self.conv1d = nn.Conv1d(
            in_channels=hidden_dim, out_channels=32, kernel_size=3, padding=1
        )
        self.bn = nn.BatchNorm1d(32)

        # attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1), nn.Softmax(dim=1)
        )

        # final prediction layers
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)  # x, y correction

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # LSTM to capture temporal dependencies
        lstm_out, _ = self.lstm(x)

        # reshape for 1D convolution: [batch, seq_len, features] -> [batch, features, seq_len]
        lstm_out = lstm_out.permute(0, 2, 1)

        # apply convolution to extract features across the sequence
        conv_out = self.relu(self.bn(self.conv1d(lstm_out)))

        # reshape back: [batch, features, seq_len] -> [batch, seq_len, features]
        conv_out = conv_out.permute(0, 2, 1)

        # compute attention weights
        attention_weights = self.attention(conv_out)

        # apply attention to get context vector
        context = torch.sum(attention_weights * conv_out, dim=1)

        # final prediction
        x = self.dropout(self.relu(self.fc1(context)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


input_dim = X_train_seq.shape[2]  # num of features
model = HybridPositioningModel(input_dim=input_dim)
loss_fn = nn.HuberLoss(delta=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, verbose=True
)

# early stop
early_stopping_patience = 15
best_val_loss = float("inf")
early_stopping_counter = 0


def train_model(
    model, train_loader, test_loader, optimizer, loss_fn, scheduler, epochs=100
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model.to(device)

    train_losses = []
    val_losses = []
    best_model_state = None
    best_val_loss = float("inf")
    early_stopping_counter = 0

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # fwd pass
            optimizer.zero_grad()
            predictions = model(batch_x)

            loss = loss_fn(predictions, batch_y)

            # backpropagation
            loss.backward()

            # gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = model(batch_x)
                loss = loss_fn(predictions, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        avg_val_loss = val_loss / len(test_loader.dataset)
        val_losses.append(avg_val_loss)

        # learning rate scheduling
        scheduler.step(avg_val_loss)

        # early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        # print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Time: {elapsed_time:.2f}s"
            )

    # load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.semilogy(train_losses, label="Training Loss")
    plt.semilogy(val_losses, label="Validation Loss")
    plt.title("Loss (Log Scale)")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("outputs/training_curve.png")

    return train_losses, val_losses


def evaluate_model(
    model, X_test_seq_tensor, y_test_seq_tensor, gnss_test, true_test, scaler_y
):
    """Comprehensive model evaluation"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        # get model predictions
        X_test_device = X_test_seq_tensor.to(device)
        predictions = model(X_test_device).cpu().numpy()

        # inverse transform predictions (correction values)
        predicted_corrections = scaler_y.inverse_transform(predictions)

        # add corrections to GNSS to get corrected positions
        # NOTE: we need to align the array lengths because of sequence creation
        offset = seq_length - 1
        corrected_positions = (
            gnss_test[offset : offset + len(predictions)] + predicted_corrections
        )
        true_positions = true_test[offset : offset + len(predictions)]
        gnss_positions = gnss_test[offset : offset + len(predictions)]

        # calculate errors
        gnss_errors = np.sqrt(np.sum((gnss_positions - true_positions) ** 2, axis=1))
        corrected_errors = np.sqrt(
            np.sum((corrected_positions - true_positions) ** 2, axis=1)
        )

        # calculate statistics
        mean_gnss_error = np.mean(gnss_errors)
        median_gnss_error = np.median(gnss_errors)
        mean_corrected_error = np.mean(corrected_errors)
        median_corrected_error = np.median(corrected_errors)
        # calculate RMSE
        rmse_gnss = np.sqrt(np.mean(gnss_errors**2))
        rmse_corrected = np.sqrt(np.mean(corrected_errors**2))
        improvement_percentage = 100 * (1 - mean_corrected_error / mean_gnss_error)
        rmse_improvement_percentage = 100 * (1 - rmse_corrected / rmse_gnss)

        # Print results
        print("\nModel Evaluation Results:")
        print(
            f"Original GNSS Error: Mean = {mean_gnss_error:.2f}m, Median = {median_gnss_error:.2f}m, RMSE = {rmse_gnss:.2f}m"
        )
        print(
            f"Corrected Error: Mean = {mean_corrected_error:.2f}m, Median = {median_corrected_error:.2f}m, RMSE = {rmse_corrected:.2f}m"
        )
        print(
            f"Improvement: Mean = {improvement_percentage:.2f}%, RMSE = {rmse_improvement_percentage:.2f}%"
        )

        # roiginal vs corrected trajectory
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.scatter(
            true_positions[:, 0],
            true_positions[:, 1],
            s=5,
            label="True Positions",
            alpha=0.8,
        )
        plt.scatter(
            gnss_positions[:, 0],
            gnss_positions[:, 1],
            s=5,
            label="GNSS Positions",
            alpha=0.5,
        )
        plt.scatter(
            corrected_positions[:, 0],
            corrected_positions[:, 1],
            s=5,
            label="Corrected Positions",
            alpha=0.5,
        )
        plt.legend()
        plt.title("Position Comparison")
        plt.axis("equal")

        # error histogram
        plt.subplot(2, 2, 2)
        plt.hist(gnss_errors, bins=50, alpha=0.5, label="GNSS Error", density=True)
        plt.hist(
            corrected_errors, bins=50, alpha=0.5, label="Corrected Error", density=True
        )
        plt.xlabel("Error (m)")
        plt.ylabel("Frequency")
        plt.title("Error Distribution")
        plt.legend()

        # error over time
        plt.subplot(2, 2, 3)
        plt.plot(gnss_errors, label="GNSS Error", alpha=0.5)
        plt.plot(corrected_errors, label="Corrected Error", alpha=0.8)
        plt.xlabel("Time Step")
        plt.ylabel("Error (m)")
        plt.title("Error Over Time")
        plt.legend()

        # error improvement histogram
        plt.subplot(2, 2, 4)
        improvement = gnss_errors - corrected_errors
        plt.hist(improvement, bins=50)
        plt.xlabel("Error Improvement (m)")
        plt.ylabel("Frequency")
        plt.title("Error Improvement Distribution")

        plt.tight_layout()
        plt.savefig("outputs/model_evaluation.png")

        # detailed trajectory section
        plt.figure(figsize=(12, 8))
        section_start = min(500, len(true_positions) - 100)
        section_end = min(section_start + 100, len(true_positions))

        plt.plot(
            true_positions[section_start:section_end, 0],
            true_positions[section_start:section_end, 1],
            "g-",
            linewidth=2,
            label="True Trajectory",
        )
        plt.plot(
            gnss_positions[section_start:section_end, 0],
            gnss_positions[section_start:section_end, 1],
            "r--",
            linewidth=1,
            label="GNSS Trajectory",
        )
        plt.plot(
            corrected_positions[section_start:section_end, 0],
            corrected_positions[section_start:section_end, 1],
            "b-",
            linewidth=1,
            label="Corrected Trajectory",
        )
        plt.legend()
        plt.title(f"Trajectory Section (Steps {section_start}-{section_end})")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.savefig("outputs/trajectory_section.png")

        results_df = pd.DataFrame(
            {
                "true_x": true_positions[:, 0],
                "true_y": true_positions[:, 1],
                "gnss_x": gnss_positions[:, 0],
                "gnss_y": gnss_positions[:, 1],
                "corrected_x": corrected_positions[:, 0],
                "corrected_y": corrected_positions[:, 1],
                "gnss_error": gnss_errors,
                "corrected_error": corrected_errors,
                "improvement": improvement,
            }
        )
        results_df.to_csv("outputs/results.csv", index=False)

        # add a dedicated plot comparing true, GNSS, and corrected paths
        plt.figure(figsize=(12, 10))
        plt.plot(
            true_positions[:, 0],
            true_positions[:, 1],
            "g-",
            linewidth=2,
            label="True Path",
        )
        plt.plot(
            gnss_positions[:, 0],
            gnss_positions[:, 1],
            "r--",
            linewidth=1.5,
            label="GNSS Path",
        )
        plt.plot(
            corrected_positions[:, 0],
            corrected_positions[:, 1],
            "b-",
            linewidth=1.5,
            label="AI Corrected Path",
        )

        # add markers to show direction
        arrow_indices = np.linspace(0, len(true_positions) - 1, 20, dtype=int)
        for i in arrow_indices:
            plt.arrow(
                true_positions[i, 0],
                true_positions[i, 1],
                (
                    true_positions[min(i + 5, len(true_positions) - 1), 0]
                    - true_positions[i, 0]
                )
                / 5,
                (
                    true_positions[min(i + 5, len(true_positions) - 1), 1]
                    - true_positions[i, 1]
                )
                / 5,
                head_width=2,
                head_length=3,
                fc="g",
                ec="g",
                alpha=0.6,
            )

        plt.title("Complete Path Comparison", fontsize=16)
        plt.xlabel("X Position (m)", fontsize=14)
        plt.ylabel("Y Position (m)", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig("outputs/path_comparison.png", dpi=300)

        return {
            "mean_gnss_error": mean_gnss_error,
            "median_gnss_error": median_gnss_error,
            "mean_corrected_error": mean_corrected_error,
            "median_corrected_error": median_corrected_error,
            "rmse_gnss": rmse_gnss,
            "rmse_corrected": rmse_corrected,
            "improvement_percentage": improvement_percentage,
            "rmse_improvement_percentage": rmse_improvement_percentage,
        }


def main():
    # Train the model
    print("Starting model training...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, optimizer, loss_fn, scheduler, epochs=150
    )

    # Evaluate the model and get metrics
    print("Evaluating model...")
    metrics = evaluate_model(
        model, X_test_seq_tensor, y_test_seq_tensor, gnss_test, true_test, scaler_y
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "input_dim": input_dim,
            "seq_length": seq_length,
            "metrics": metrics,
        },
        "outputs/positioning_model.pt",
    )


if __name__ == "__main__":
    main()
