import torch
import torch.nn as nn
import torch.nn.functional as F
# from .backbone import resnet18, resnet34, resnet101
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion
from transformers import AutoModel


class AClassifier(nn.Module):
    def __init__(self, args):
        super(AClassifier, self).__init__()
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        self.net = resnet18(modality='audio')
        self.classifier = nn.Linear(args.embed_dim, n_classes)

    def forward(self, audio):
        a = self.net(audio)
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)
        out = self.classifier(a)
        return out


class VClassifier(nn.Module):
    def __init__(self, args):
        super(VClassifier, self).__init__()
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        self.net = resnet18(modality='visual')
        self.classifier = nn.Linear(args.embed_dim, n_classes)

    def forward(self, visual, B):
        v = self.net(visual)
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        out = self.classifier(v)
        return out


class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

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

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

    def forward(self, audio, visual):

        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a, v, out = self.fusion_module(a, v)

        return a, v, out


class AVClassifier_34(nn.Module):
    def __init__(self, args):
        super(AVClassifier_34, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

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

        self.audio_net = resnet34(modality='audio')
        self.visual_net = resnet34(modality='visual')

    def forward(self, audio, visual):

        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        # print('concat: ', v.shape)
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        # print('dis: ', v.shape)
        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a, v, out = self.fusion_module(a, v)

        return a, v, out


class AVClassifier_101(nn.Module):
    def __init__(self, args):
        super(AVClassifier_101, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

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

        self.audio_net = resnet101(modality='audio')
        self.visual_net = resnet101(modality='visual')

    def forward(self, audio, visual):

        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        # print('concat: ', v.shape)
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        # print('dis: ', v.shape)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)
        # print('avg: ', v.shape)
        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a, v, out = self.fusion_module(a, v)

        return a, v, out


class CLClassifier(nn.Module):
    def __init__(self, args):
        super(CLClassifier, self).__init__()

        self.fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if self.fusion == 'concat':
            self.fc_out = nn.Linear(args.embed_dim * 2, n_classes)
        elif self.fusion == 'sum':
            self.fc_x = nn.Linear(args.embed_dim, n_classes)
            self.fc_y = nn.Linear(args.embed_dim, n_classes)

    def forward(self, x, y):
        if self.fusion == 'concat':
            output = torch.cat((x, y), dim=1)
            output = self.fc_out(output)
        return output


# Colored-and-gray-MNIST
class convnet(nn.Module):
    def __init__(self, num_classes=10, modal='gray'):
        super(convnet, self).__init__()

        self.modal = modal

        if modal == 'gray':
            in_channel = 1
        elif modal == 'colored':
            in_channel = 3
        else:
            raise ValueError('non exist modal')
        self.bn0 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(64, 512)

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu(x)  # 28x28
        x = self.maxpool(x)  # 14x14

        x = self.conv2(x)
        x = self.relu(x)  # 14x14
        x = self.conv3(x)
        x = self.relu(x)  # 7x7
        x = self.conv4(x)
        x = self.relu(x)  # 7x7

        feat = x
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        feat = self.fc(feat)

        return feat


class CGClassifier(nn.Module):
    def __init__(self, args):
        super(CGClassifier, self).__init__()

        fusion = args.fusion_method

        n_classes = 10

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

        self.gray_net = convnet(modal='gray')
        self.colored_net = convnet(modal='colored')

    def forward(self, gray, colored):
        g = self.gray_net(gray)
        c = self.colored_net(colored)

        g = torch.flatten(g, 1)
        c = torch.flatten(c, 1)

        g, c, out = self.fusion_module(g, c)
        return g, c, out


class GrayClassifier(nn.Module):
    def __init__(self, args):
        super(GrayClassifier, self).__init__()
        if args.dataset == 'CGMNIST':
            n_classes = 10

        self.net = convnet(modal='gray')
        self.classifier = nn.Linear(args.embed_dim, n_classes)

    def forward(self, gray):
        g = self.net(gray)
        g = torch.flatten(g, 1)
        g_out = self.classifier(g)
        return g_out


class ColoredClassifier(nn.Module):
    def __init__(self, args):
        super(ColoredClassifier, self).__init__()
        if args.dataset == 'CGMNIST':
            n_classes = 10

        self.net = convnet(modal='colored')
        self.classifier = nn.Linear(args.embed_dim, n_classes)

    def forward(self, color):
        c = self.net(color)
        c = torch.flatten(c, 1)
        c_out = self.classifier(c)
        return c_out


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
    
from MPLMM.src.model import PromptModel
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
        
        # 4. 单模态分类头 (PMR 算法需要计算单模态的 score 来评估学习进度)
        self.classifier_t = nn.Linear(hyp_params.d_l, n_classes)
        self.classifier_c = nn.Linear(hyp_params.d_a, n_classes)

    def forward(self, text_input_ids, text_attn_mask, code_input_ids, code_attn_mask, missing_mod):
        # 1. 提取全序列特征 [Batch, Seq_len, 768]
        t_seq = self.text_net(input_ids=text_input_ids, attention_mask=text_attn_mask).last_hidden_state[:, 0:1, :]
        c_seq = self.code_net(input_ids=code_input_ids, attention_mask=code_attn_mask).last_hidden_state[:, 0:1, :]
        
        # 2. 维度映射
        t_seq = self.text_proj(t_seq)
        c_seq = self.code_proj(c_seq)
        
        # 3. 送入 PromptModel 获取独立特征和总预测
        feat_t, feat_c, out = self.prompt_model(t_seq, c_seq, missing_mod)
        
        # 4. 计算单模态预测 (专供 PMR 算法计算比例使用)
        out_t = self.classifier_t(feat_t)
        out_c = self.classifier_c(feat_c)
        
        return feat_t, feat_c, out, out_t, out_c