import torch
from torch.utils.data import Dataset
import pickle
import random

class CSVTextCodeData(Dataset):
    def __init__(self, data_path, split, drop_rate, full_data=False):
        super(CSVTextCodeData, self).__init__()
        
        # 1. 直接加载我们离线提取好的 .pkl 特征文件
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
            
        self.data = data[split]  # split 为 'train', 'valid' 或 'test'
        self.split = split
        self.drop_rate = drop_rate
        self.full_data = full_data
        
        # 2. 自动获取维度和长度 (从张量 shape 中读取)
        # text 的 shape 是 [样本数, 512, 768]，audio(代码) 的 shape 是 [样本数, 512, 768]
        self.orig_dims = [
            self.data['text'][0].shape[1],   # 768
            self.data['audio'][0].shape[1]   # 768
        ]
        self.seq_lens = [
            self.data['text'][0].shape[0],   # 512
            self.data['audio'][0].shape[0]   # 512
        ]

    def get_dim(self):
        return self.orig_dims
    
    def get_seq_len(self):
        return self.seq_lens

    def __len__(self):
        return self.data['labels'].shape[0]
    
    def get_missing_mode(self):
        """
        双模态缺失逻辑：
        0: 缺描述文本
        1: 缺代码
        2: 都不缺失
        """
        if self.full_data:
            return 2
        if random.random() < self.drop_rate:
            return random.randint(0, 1)
        else:
            return 2

    def __getitem__(self, idx):
        # 3. 直接取出 Numpy 数组并转为 PyTorch 张量，速度极快
        L_feat = torch.tensor(self.data['text'][idx]).float()
        C_feat = torch.tensor(self.data['audio'][idx]).float()
        label = torch.tensor(self.data['labels'][idx]).float()
        
        X = (L_feat, C_feat)
        # --- 新加逻辑 ---
        # 检查文本向量是不是全 0 (或者能量极低)
        real_text_missing = (torch.sum(torch.abs(L_feat)) < 1e-6)
        # 检查代码向量是不是全 0
        real_code_missing = (torch.sum(torch.abs(C_feat)) < 1e-6)
        
        # if real_text_missing:
        #     missing_code = 0  
        # elif real_code_missing:
        #     missing_code = 1
        # else:
        #     missing_code = 2
        missing_code = self.get_missing_mode()

        return X, label, missing_code