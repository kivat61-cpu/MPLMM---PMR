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
        fqn = str(row.get('FQN', ''))

        # 判断是否为真实的缺失 (比如 pandas 读进来是 NaN，或者空字符串)
        real_text_missing = pd.isna(raw_text) or str(raw_text).strip() == ""
        real_code_missing = pd.isna(raw_code) or str(raw_code).strip() == ""

        if real_text_missing:
            missing_mode = 0  # 真实缺文本
        elif real_code_missing:
            missing_mode = 1  # 真实缺代码
        else:
            # 走到这里说明真实数据都不缺，我们再决定要不要“模拟缺失”来锻炼模型
            if self.full_data:
                missing_mode = 2
            elif random.random() < self.drop_rate:
                missing_mode = random.randint(0, 1)  # 假装缺失一个
            else:
                missing_mode = 2  # 正常完整使用

        # API 标识前缀：只加到不缺失的模态上
        api_prefix = f"[API: {fqn}] " if fqn and fqn.strip() and fqn != 'nan' else ""

        if missing_mode == 0:
            text_str = "[PAD]"  # 文本真实缺失，不能加 API 名
            code_str = api_prefix + str(raw_code)
        elif missing_mode == 1:
            text_str = api_prefix + str(raw_text)
            code_str = "[PAD]"  # 代码真实缺失，不能加 API 名
        else:
            text_str = api_prefix + str(raw_text)
            code_str = api_prefix + str(raw_code)

        text_enc = self.text_tokenizer(
            text_str, truncation=True, padding='max_length', 
            max_length=self.max_length, return_tensors='pt'
        )
        
        code_enc = self.code_tokenizer(
            code_str, truncation=True, padding='max_length', 
            max_length=self.max_length, return_tensors='pt'
        )

        

        return {
            'text_input_ids': text_enc['input_ids'].squeeze(0),
            'text_attention_mask': text_enc['attention_mask'].squeeze(0),
            'code_input_ids': code_enc['input_ids'].squeeze(0),
            'code_attention_mask': code_enc['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'missing_mode': torch.tensor(missing_mode, dtype=torch.long) # DataLoader会自动将它打包成一维 Tensor
        }