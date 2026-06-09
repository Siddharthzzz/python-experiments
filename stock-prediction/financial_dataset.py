import torch 
from torch.utils.data import Dataset
import pandas as pd 
import numpy as np
class FinancialDataset(Dataset):
    def __init__(self,df,feature_cols,target_col='returns',seq_len=20):
        
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        
        features = df[feature_cols].values
        targets = df[target_col].values
        X_list , y_list = [],[]
        
        for i in range(seq_len,len(df) -1):
            X_list.append(features[i - seq_len:i])
            y_list.append(1.0 if targets[i+1] > 0 else 0.0)
        
        self.X = torch.tensor(np.array(X_list),dtype=torch.float32)
        self.y = torch.tensor(y_list,dtype=torch.float32).unsqueeze(dim=1)
        
    def __len__(self):
            return len(self.X)
        
    def __getitem__(self,idx):
            return self.X[idx],self.y[idx]      