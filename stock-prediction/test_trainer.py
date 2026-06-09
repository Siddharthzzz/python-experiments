import torch
from torch.utils.data import DataLoader,TensorDataset
from Trainer import Trainer
from model import TinyStockPredictor

X = torch.randn(200,60)
y = (X[:,0] + X[:,1] > 0).float().unsqueeze(1)

dataset = TensorDataset(X,y)
train_ds = TensorDataset(X[:150], y[:150])
val_ds = TensorDataset(X[150:], y[150:])

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32)

model = TinyStockPredictor(input_size=60)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn = torch.nn.BCEWithLogitsLoss()

trainer = Trainer(model,optimizer,loss_fn)
trainer.fit(train_dl,val_dl,epochs=5)