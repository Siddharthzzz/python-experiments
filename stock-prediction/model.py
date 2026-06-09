import torch.nn as nn 

class TinyStockPredictor(nn.Module):
    def __init__(self,input_size = 60,hidden_size = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,1)
        )
    def forward(self,x):
        if x.dim() == 3:
            x = x.view(x.size(0),-1)
        return self.net(x)
    
    