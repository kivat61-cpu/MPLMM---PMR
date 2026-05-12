import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion
from transformers import AutoModel


class TextCodeClassifier(nn.Module):
    def __init__(self, args, n_classes=2): # 这里的 n_classes 根据您的数据集修改
        super(TextCodeClassifier, self).__init__()

        # 1. 完整保留原项目的融合模块逻辑
        fusion = args.fusion_method
        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        # 2. 替换 Backbone：实例化文本与代码分支
        # 文本分支使用 Sentence-BERT 提取高质量语义特征
        self.text_net = AutoModel.from_pretrained('bert-base-uncased')
        # 代码分支使用 CodeBERT 提取代码逻辑特征
        self.code_net = AutoModel.from_pretrained('microsoft/codebert-base')

        # 3. 增加维度对齐层 (Projection Layer)
        # BERT模型通常输出 384 或 768 维，而原项目的 fusion_module 默认接收 args.embed_dim (如 512维)
        self.text_proj = nn.Linear(768, args.embed_dim) # MiniLM-L6 是 384 维
        self.code_proj = nn.Linear(768, args.embed_dim) # CodeBERT 是 768 维

    def forward(self, text_input_ids, text_attn_mask, code_input_ids, code_attn_mask):
        # 1. 文本分支：提取特征向量
        text_outputs = self.text_net(input_ids=text_input_ids, attention_mask=text_attn_mask)
        # 取 [CLS] token 的向量作为整个文本序列的特征
        t = text_outputs.last_hidden_state[:, 0, :] 
        
        # 2. 代码分支：提取特征向量
        code_outputs = self.code_net(input_ids=code_input_ids, attention_mask=code_attn_mask)
        # 同样取 [CLS] token
        c = code_outputs.last_hidden_state[:, 0, :] 

        # 3. 维度对齐映射 (投射到 512 维)
        t = self.text_proj(t)
        c = self.code_proj(c)

        # 4. 进入原项目的多模态融合层
        # 这里 t 相当于原来的 a(音频), c 相当于原来的 v(视觉)
        t, c, out = self.fusion_module(t, c)

        # 返回文本纯预测、代码纯预测、以及融合总预测 (完美契合 OGM/PMR 策略)
        return t, c, out
    
from MPLMM.src.model import MULTModel, PromptModel
class PromptTextCodeClassifier(nn.Module):
    def __init__(self, args, hyp_params, n_classes=2):
        super(PromptTextCodeClassifier, self).__init__()
        
        # 1. 底层特征提取器
        self.text_net = AutoModel.from_pretrained('bert-base-uncased')
        self.code_net = AutoModel.from_pretrained('microsoft/codebert-base')
        
        # 2. 维度对齐 (将 768 维映射到 PromptModel 需要的维度，假设为 hyp_params.orig_d_l)
        self.text_proj = nn.Linear(768, hyp_params.orig_d_l)
        self.code_proj = nn.Linear(768, hyp_params.orig_d_a)
        
        # 3. 核心机制：替换掉原本简单的 fusion_module，使用你的 PromptModel
        self.prompt_model = PromptModel(hyp_params)

        self.num_tokens = hyp_params.seq_len[0]

        # 4. 单模态分类头 (PMR 算法需要计算单模态的 score 来评估学习进度)
        self.classifier_t = nn.Linear(hyp_params.d_l, n_classes)
        self.classifier_c = nn.Linear(hyp_params.d_a, n_classes)

    def forward(self, text_input_ids, text_attn_mask, code_input_ids, code_attn_mask, missing_mod):
        num_tokens = self.num_tokens

        # 1. 提取特征序列
        full_t = self.text_net(input_ids=text_input_ids, attention_mask=text_attn_mask).last_hidden_state
        full_c = self.code_net(input_ids=code_input_ids, attention_mask=code_attn_mask).last_hidden_state
        if num_tokens == 1:
            t_seq = full_t[:, 0:1, :]   # [CLS] token
            c_seq = full_c[:, 0:1, :]
        else:
            t_seq = F.adaptive_avg_pool1d(full_t.transpose(1, 2), num_tokens).transpose(1, 2)
            c_seq = F.adaptive_avg_pool1d(full_c.transpose(1, 2), num_tokens).transpose(1, 2)

        # 2. 维度映射
        t_seq = self.text_proj(t_seq)
        c_seq = self.code_proj(c_seq)

        # 3. 送入 PromptModel 获取独立特征和总预测
        feat_t, feat_c, out = self.prompt_model(t_seq, c_seq, missing_mod)

        # 4. 计算单模态预测 (专供 PMR 算法计算比例使用)
        out_t = self.classifier_t(feat_t)
        out_c = self.classifier_c(feat_c)

        return feat_t, feat_c, out, out_t, out_c


class MULTTextCodeClassifier(nn.Module):
    def __init__(self, args, hyp_params, n_classes=2):
        super(MULTTextCodeClassifier, self).__init__()

        self.text_net = AutoModel.from_pretrained('bert-base-uncased')
        self.code_net = AutoModel.from_pretrained('microsoft/codebert-base')

        self.text_proj = nn.Linear(768, hyp_params.orig_d_l)
        self.code_proj = nn.Linear(768, hyp_params.orig_d_a)

        self.mult_model = MULTModel(hyp_params)

        self.classifier_t = nn.Linear(hyp_params.d_l, n_classes)
        self.classifier_c = nn.Linear(hyp_params.d_a, n_classes)

        self.num_tokens = hyp_params.seq_len[0]

    def forward(self, text_input_ids, text_attn_mask, code_input_ids, code_attn_mask, missing_mod=None):
        num_tokens = self.num_tokens

        full_t = self.text_net(input_ids=text_input_ids, attention_mask=text_attn_mask).last_hidden_state
        full_c = self.code_net(input_ids=code_input_ids, attention_mask=code_attn_mask).last_hidden_state
        if num_tokens == 1:
            t_seq = full_t[:, 0:1, :]   # [CLS] token
            c_seq = full_c[:, 0:1, :]
        else:
            t_seq = F.adaptive_avg_pool1d(full_t.transpose(1, 2), num_tokens).transpose(1, 2)
            c_seq = F.adaptive_avg_pool1d(full_c.transpose(1, 2), num_tokens).transpose(1, 2)

        t_seq = self.text_proj(t_seq)
        c_seq = self.code_proj(c_seq)

        feat_t, feat_c, out = self.mult_model(t_seq, c_seq)

        # feat_t/feat_c: [N, B, D] → mean over tokens → [B, D]
        feat_t = feat_t.mean(dim=0)
        feat_c = feat_c.mean(dim=0)

        out_t = self.classifier_t(feat_t)
        out_c = self.classifier_c(feat_c)

        return feat_t, feat_c, out, out_t, out_c