import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from stock_loader import stock_data_loader     # copied from day02
from financial_dataset import FinancialDataset
from Trainer import Trainer                   # built earlier
from model import TinyStockPredictor          # built earlier

# --------------------------
# 1. Get real NVDA data
# --------------------------
loader = stock_data_loader("NVDA", start_date="2023-01-01")
loader.download_data()
loader.Clean_data()
loader.add_features()

feature_cols = ['returns', 'ma_20', 'volatility']   # you can later add 'rsi'
dataset = FinancialDataset(loader.data, feature_cols, seq_len=20)

# --------------------------
# 2. Chronological split (no future leakage)
# --------------------------
n = len(dataset)
val_start = int(n * 0.8)          # first 80% train
test_start = int(n * 0.9)         # next 10% val, last 10% test

train_ds = Subset(dataset, range(0, val_start))
val_ds   = Subset(dataset, range(val_start, test_start))
test_ds  = Subset(dataset, range(test_start, n))

print(f"Samples → Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=32, shuffle=False)
test_dl  = DataLoader(test_ds, batch_size=32, shuffle=False)

# --------------------------
# 3. Model, optimizer, loss
# --------------------------
input_size = 20 * len(feature_cols)   # 20 days × 3 features = 60
model = TinyStockPredictor(input_size=input_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()       # expects raw logits

# --------------------------
# 4. Train
# --------------------------
trainer = Trainer(model, optimizer, loss_fn)
trainer.fit(train_dl, val_dl, epochs=10)

# --------------------------
# 5. Evaluate on unseen future data
# --------------------------
test_loss, test_acc = trainer.evaluate(test_dl)
print(f"\n=== Test Set (future data) ===")
print(f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

# --------------------------
# 6. Predict next trading day (tomorrow)
# --------------------------
model.eval()
last_20 = loader.data[feature_cols].values[-20:]    # most recent 20 days
X_next = torch.tensor(last_20, dtype=torch.float32).unsqueeze(0)  # (1,20,3)
X_next = X_next.reshape(1, -1)   # reshape works for non-contiguous tensors                       # flatten to (1,60)

with torch.no_grad():
    logit = model(X_next)
    prob = torch.sigmoid(logit).item()
    direction = "UP" if prob > 0.5 else "DOWN"
    print(f"\nNext day forecast: {direction} (probability = {prob:.4f})")