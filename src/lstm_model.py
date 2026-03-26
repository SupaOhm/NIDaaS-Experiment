"""
lstm_model.py
-------------
NIDSaaS-DDA Architecture: LSTM-based Anomaly Forecasting.
Implemented with PyTorch following Section III-B-3 of the paper.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class _LSTMNet(nn.Module):
    def __init__(self, input_size, lstm_units, dropout):
        super().__init__()
        # Section III-B-3-2: Base feature vector undergoes Batch Normalization
        self.bn1 = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(input_size, lstm_units, num_layers=2, 
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(lstm_units, input_size) # Predicting NEXT step feature vector
        # Section III-B-3-3: Dropout layer Integrated (0.2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (batch, seq, feat)
        b, s, f = x.shape
        # Normalize features across batch/seq
        x_flat = x.view(b * s, f)
        x_norm = self.bn1(x_flat)
        x = x_norm.view(b, s, f)
        
        out, _ = self.lstm(x)
        # We take the output of the LAST hidden state to predict the NEXT vector
        out = self.fc(out[:, -1, :])
        return out

class LSTMModel:
    def __init__(self, input_shape, lstm_units=64, dropout=0.2, learning_rate=1e-3):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout = 0.2
        self.learning_rate = learning_rate
        self.device = torch.device("cpu")
        print(f"[lstm_model] NIDSaaS-DDA Architecture (CPU-Mode): {self.device}")
        
        self.net = _LSTMNet(input_shape[1], lstm_units, 0.2).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        # Forecasting task uses MSE (Mean Squared Error) for training
        self.criterion = nn.MSELoss()

    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=256, patience=5):
        # NOTE: For anomaly forecasting, 'y_train' is actually the NEXT vector in sequence.
        # But to keep API simple, we expect X_train to be sequences [t-seq:t]
        # and y_train to be the target vector at [t+1].
        
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_v = torch.tensor(X_val,   dtype=torch.float32).to(self.device)
        y_v = torch.tensor(y_val,   dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        no_improve = 0
        best_state = None

        print(f"[lstm_model] Training Forecasting Engine (Epochs={epochs})...")
        for epoch in range(1, epochs + 1):
            self.net.train()
            epoch_loss = 0.0
            for Xb, yb in loader:
                self.optimizer.zero_grad()
                preds = self.net(Xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(Xb)
            
            epoch_loss /= len(X_t)
            self.net.eval()
            with torch.no_grad():
                val_preds = self.net(X_v)
                val_loss = self.criterion(val_preds, y_v).item()
            
            if epoch % 5 == 0 or epoch == 1:
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

        if best_state:
            self.net.load_state_dict(best_state)
        return self

    def predict_anomaly_score(self, X, y_actual):
        """
        Section III-B-3-3: Anomaly score d = Euclidean distance(actual, predicted).
        """
        self.net.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_actual, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.net(X_t)
            # Euclidean distance: sqrt(sum((p - a)^2))
            # d = torch.sqrt(torch.sum((preds - y_t)**2, dim=1))
            # But the paper says d(v, hat_v). We'll use the dim=1 norm.
            d = torch.norm(preds - y_t, p=2, dim=1)
            return d.cpu().numpy()

    def predict(self, X, y_actual, threshold=0.1):
        """Binary prediction based on threshold."""
        scores = self.predict_anomaly_score(X, y_actual)
        return (scores >= threshold).astype(int)

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({"net": self.net.state_dict(), "meta": {"units": self.lstm_units}}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt["net"])
        return self
