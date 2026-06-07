import torch
from torch.utils.data import DataLoader,TensorDataset
from Trainer import Trainer

X = torch.randn(200,10)
y = (X[:,0] + X[:,1] > 0).long()

dataset = TensorDataset(X,y)
train_ds = TensorDataset(X[:150], y[:150])
val_ds = TensorDataset(X[150:], y[150:])

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32)

model = torch.nn.Linear(10,2)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

trainer = Trainer(model,optimizer,loss_fn)
trainer.fit(train_dl,val_dl,epochs=5)