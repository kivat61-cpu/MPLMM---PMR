import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class TextCodeDataset(Dataset):
    def __init__(self, dataframe, max_length=512):
        # 接收切分好的 DataFrame 并重置索引，防止错乱
        self.data = dataframe.reset_index(drop=True) 
        
        # 统一使用基础 BERT 处理文本和代码（后续可无缝切换 CodeBERT）
        self.text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.code_tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 读取您指定的列名
        text_str = str(row['text'])
        code_str = str(row['code'])
        label = int(row['label'])

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
            'label': torch.tensor(label, dtype=torch.long)
        }