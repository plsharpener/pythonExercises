import torch
from torch.utils.data import Dataset
import pandas as pd

class BulldozerDataset(Dataset):
    """数据集演示"""
    def __init__(self,csv_file):
        self.df = pd.read_csv(csv_file)
    
    def __len__(self):
        """返回df的长度"""
        return len(self.df)

    def __getitem__(self, idx):
        """根据 idx 返回一行数据"""
        return self.df.iloc[idx].SalePrice

# ds_demo = BulldozerDataset('./test.csv')
# print(len(ds_demo))