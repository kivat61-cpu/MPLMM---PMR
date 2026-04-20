import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random
import pandas as pd

class TextCodeDataset(Dataset):
    def __init__(self, dataframe, max_length=512, drop_rate=0.2, full_data=False):
        # 接收切分好的 DataFrame 并重置索引，防止错乱
        self.data = dataframe.reset_index(drop=True) 
        
        # 使用基础BERT处理文本，CodeBERT处理代码
        self.text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.code_tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        self.max_length = max_length

        self.drop_rate = drop_rate
        self.full_data = full_data

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 读取您指定的列名
        # text_str = str(row['text'])
        # code_str = str(row['code'])
        raw_text = row['text']
        raw_code = row['code']
        label = int(row['label'])

        # 判断是否为真实的缺失 (比如 pandas 读进来是 NaN，或者空字符串)
        real_text_missing = pd.isna(raw_text) or str(raw_text).strip() == ""
        real_code_missing = pd.isna(raw_code) or str(raw_code).strip() == ""

        # 为了防止 tokenizer 报错，如果真实缺失，我们塞入一个毫无意义的占位符（比如 [PAD]）
        # 反正底层模型看到 missing_mode=0/1 时，会直接丢弃这个模态的特征去生成它
        text_str = "[PAD]" if real_text_missing else str(raw_text)
        code_str = "[PAD]" if real_code_missing else str(raw_code)

        text_enc = self.text_tokenizer(
            text_str, truncation=True, padding='max_length', 
            max_length=self.max_length, return_tensors='pt'
        )
        
        code_enc = self.code_tokenizer(
            code_str, truncation=True, padding='max_length', 
            max_length=self.max_length, return_tensors='pt'
        )

        if real_text_missing:
            missing_mode = 0  # 真实缺文本
        elif real_code_missing:
            missing_mode = 1  # 真实缺代码
        else:
            # 走到这里说明真实数据都不缺，我们再决定要不要“模拟缺失”来锻炼模型
            if self.full_data:
                missing_mode = 2  # 考试模式，有啥用啥，绝不假装缺失
            elif random.random() < self.drop_rate:
                missing_mode = random.randint(0, 1) # 训练模式，假装缺失一个
            else:
                missing_mode = 2  # 正常完整使用

        return {
            'text_input_ids': text_enc['input_ids'].squeeze(0),
            'text_attention_mask': text_enc['attention_mask'].squeeze(0),
            'code_input_ids': code_enc['input_ids'].squeeze(0),
            'code_attention_mask': code_enc['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'missing_mode': torch.tensor(missing_mode, dtype=torch.long) # DataLoader会自动将它打包成一维 Tensor
        }