import torch

class Trainer():
    def __init__(self,model,optimizer,loss_fn,device="cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
    def train_epoch(self,dataloader):
        self.model.train()
        total_loss = 0.0
        for X,y in dataloader:
            X,y = X.to(self.device),y.to(self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(X)
            loss = self.loss_fn(preds,y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()*X.size(0)
            return total_loss / len(dataloader.dataset)
    def evaluate(self,dataloader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        with torch.no_grad():
            for X,y in dataloader:
                X,y = X.to(self.device),y.to(self.device)
                preds = self.model(X)
                loss = self.loss_fn(preds,y)
                total_loss += loss.item()*X.size(0)
                if preds.shape[1] == 1:
                    predicted = (torch.sigmoid(preds) > 0.5).long().squeeze()
                else:
                    predicted = preds.argmax(dim=1)
                    correct += (predicted == y).sum().item()
        avg_loss = total_loss / len(dataloader.dataset)
        accuracy = correct / len(dataloader.dataset)
        return avg_loss , accuracy
    def fit(self,train_dl,val_dl,epochs=10):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_dl)
            val_loss,val_acc = self.evaluate(val_dl)
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
            
                    
                
            
        
        
        
        
            
            
        
        
        