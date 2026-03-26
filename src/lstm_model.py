"""
lstm_model.py
-------------
Supervised LSTM sequence classifier for intrusion detection.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _LSTMClassifierNet(nn.Module):
    def __init__(self, input_size, lstm_units, dropout):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_units,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        hidden_mid = max(lstm_units // 2, 16)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_units, hidden_mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mid, 1),
        )

    def forward(self, x):
        # x: (batch, seq, feat)
        b, s, f = x.shape
        x_flat = x.reshape(b * s, f)
        x_norm = self.bn1(x_flat)
        x = x_norm.reshape(b, s, f)

        out, _ = self.lstm(x)
        out = out[:, -1, :]      # last hidden state
        logits = self.head(out)  # (batch, 1)
        return logits.squeeze(1)


class LSTMModel:
    def __init__(
        self,
        input_shape,
        lstm_units=64,
        dropout=0.2,
        learning_rate=1e-3,
        pos_weight=1.0,
    ):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.pos_weight = float(pos_weight)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[lstm_model] Supervised LSTM classifier on: {self.device}")

        self.net = _LSTMClassifierNet(
            input_shape[1],
            lstm_units,
            dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight], dtype=torch.float32, device=self.device)
        )

    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=256, patience=5):
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Empty training sequence set for LSTM.")

        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_v = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        no_improve = 0
        best_state = None

        print(f"[lstm_model] Training sequence classifier (Epochs={epochs})...")
        for epoch in range(1, epochs + 1):
            self.net.train()
            epoch_loss = 0.0

            for Xb, yb in loader:
                self.optimizer.zero_grad()
                logits = self.net(Xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(Xb)

            epoch_loss /= len(X_t)

            self.net.eval()
            with torch.no_grad():
                val_logits = self.net(X_v)
                val_loss = self.criterion(val_logits, y_v).item()

            if epoch % 2 == 0 or epoch == 1:
                print(f"  Epoch {epoch}/{epochs} | Loss: {epoch_loss:.6f} | Val: {val_loss:.6f}")

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        return self

    def predict_proba(self, X):
        self.net.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.net(X_t)
            probs = torch.sigmoid(logits)
            return probs.cpu().numpy()

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "net": self.net.state_dict(),
                "meta": {
                    "units": self.lstm_units,
                    "dropout": self.dropout,
                    "pos_weight": self.pos_weight,
                },
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt["net"])
        return self