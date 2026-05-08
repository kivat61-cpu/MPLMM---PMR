import argparse
import os
import sys
# 获取当前脚本所在目录 (PMR 文件夹)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目总根目录 (MPLMM + PMR 文件夹)
root_dir = os.path.dirname(current_dir)
# 获取 MPLMM 的文件夹路径
mplmm_dir = os.path.join(root_dir, 'MPLMM')

# 将总根目录和 MPLMM 目录都加入搜索路径
# 这样既能找到 MPLMM.src，也能让 MPLMM 内部代码找到 modules
for path in [root_dir, mplmm_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from utils.utils import setup_seed


import time


import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.TextCodeDataset import TextCodeDataset
from models.basic_model import MULTTextCodeClassifier, PromptTextCodeClassifier, TextCodeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

class MPLMMConfig:
    def __init__(self, args):
        # --- 1. 你命令中明确指定的参数 ---
        self.dataset = "csv"               # --dataset "csv"
        self.data_path = "extracted_features.pkl" # --data_path
        self.num_epochs = args.epochs               # --num_epochs 
        self.batch_size = args.batch_size               # --batch_size 
        self.drop_rate = 0.4               # --drop_rate 0.4
        self.name = "multi_binary_model.pt" # --name
        
        # --- 2. 源代码中的默认架构参数 (Architecture) ---
        self.layers = 5                    # 对应命令行的 nlevels (default: 5)
        # self.proj_dim = 30                 # 投影维度 (default: 30)
        self.proj_dim = args.embed_dim
        self.num_heads = 8                 # 注意力头数 (default: 5)
        self.attn_mask = True              # 是否使用 mask (default: True)
        self.prompt_dim = self.proj_dim    # Prompt 维度 (default: 30)
        self.prompt_length = 16            # Prompt 长度 (default: 16)
        self.d_l = self.proj_dim
        self.d_a = self.proj_dim
        # --- 3. 源代码中的默认 Dropout 参数 ---
        self.attn_dropout = 0.1
        self.attn_dropout_a = 0.1
        self.attn_dropout_v = 0.1          # 虽然双模态暂不用，但模型初始化可能需要
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0.1
        self.embed_dropout = 0.25
        
        # --- 4. 针对 BERT/CodeBERT 骨干网路自动匹配的参数 ---
        # 因为在融合模型中使用了 BERT/CodeBERT，其输出维度固定为 768
        self.orig_d_l = 768                
        self.orig_d_a = 768                
        
        # --- 5. 任务逻辑参数 (基于 main.py 中的映射逻辑) ---
        self.output_dim = 2                # dataset 为 "csv" 时固定为 2
        self.criterion = "CrossEntropyLoss" # dataset 为 "csv" 时使用交叉熵
        self.use_cuda = torch.cuda.is_available()
        
        # --- 6. 序列长度 (seq_len) ---
        # 注意：这必须和你 PMR 项目中 TextCodeDataset.py 里的 max_length 一致
        # 假设你的 Tokenizer 设置的 max_length 是 512
        self.seq_len = (1, 1)          # (llen, alen)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--model_type', default='textcode', type=str,
                        choices=['textcode', 'prompt_textcode', 'mult_textcode'],
                        help='textcode=TextCodeClassifier, prompt_textcode=PromptTextCodeClassifier, mult_textcode=MULTTextCodeClassifier')
    parser.add_argument('--modulation', default='OGM_GE', type=str,
                        choices=['Normal', 'OGM', 'OGM_GE', 'Acc', 'Proto'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--fps', default=1, type=int, help='Extract how many frames in a second')
    parser.add_argument('--num_frame', default=1, type=int, help='use how many frames for train')

    parser.add_argument('--optimizer', default='SGD', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--momentum_coef', default=0.2, type=float)
    parser.add_argument('--proto_update_freq', default=50, type=int, help='steps')

    # parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=100, type=int, help='where modulation ends')
    parser.add_argument('--alpha', default=1.0, type=float, help='alpha in Proto')

    parser.add_argument('--ckpt_path', default='ckpt', type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', action='store_true', help='whether to visualize')
    parser.add_argument('--logs_path', default='logs', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)

    parser.add_argument('--gpu', type=int, default=0)  # gpu
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--drop_rate', type=float, default=0.2, 
                        help='训练时模拟模态缺失的概率 (默认: 0.2)')
    
    parser.add_argument('--full_data', action='store_true',
                        help='如果加上此参数，训练集将强制不模拟缺失 (相当于退化为普通双模态)')

    # args = parser.parse_args()
    #
    # args.use_cuda = torch.cuda.is_available() and not args.no_cuda

    return parser.parse_args()


def EU_dist(x1, x2):
    d_matrix = torch.zeros(x1.shape[0], x2.shape[0]).to(x1.device)
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            d = torch.sqrt(torch.dot((x1[i] - x2[j]), (x1[i] - x2[j])))
            d_matrix[i, j] = d
    return d_matrix


def dot_product_angle_tensor(v1, v2):
    vector_dot_product = torch.dot(v1, v2)
    arccos = torch.acos(vector_dot_product / (torch.norm(v1, p=2) * torch.norm(v2, p=2)))
    angle = np.degrees(arccos.data.cpu().numpy())
    return arccos, angle


def grad_amplitude_diff(v1, v2):
    len_v1 = torch.norm(v1, p=2)
    len_v2 = torch.norm(v2, p=2)
    return len_v1, len_v2, len_v1 - len_v2


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler,
                text_proto, code_proto, writer=None):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.train()
    print("Start training ... ")

    _loss = 0
    _loss_t = 0
    _loss_c = 0
    _loss_p_t = 0
    _loss_p_c = 0

    _t_angle = 0
    _c_angle = 0
    _t_diff = 0
    _c_diff = 0
    _ratio_t = 0
    _ratio_t_p = 0

    # angle_file = args.logs_path + '/Method-CE-Proto-grad-amp' + '/angle-' + args.dataset + '-' + args.fusion_method + '-bsz' + \
    #              str(args.batch_size) + '-lr' + str(args.learning_rate) \
    #              + '-epoch' + str(args.epochs) + '-' + args.modulation + str(args.alpha) + \
    #              '-mon' + str(args.momentum_coef) + '-' + str(args.num_frame) + '-end' + str(args.modulation_ends) + '.txt'
    # f_angle = open(angle_file, 'a')

    # for step, (spec, image, label) in enumerate(dataloader):

    #     spec = spec.to(device)  # B x 257 x 1004(CREMAD 299)
    #     image = image.to(device)  # B x 1(image count) x 3 x 224 x 224
    #     label = label.to(device)  # B

    #     optimizer.zero_grad()

    #     # TODO: make it simpler and easier to extend
    #     if args.dataset != 'CGMNIST':
    #         a, v, out = model(spec.unsqueeze(1).float(), image.float())
    #     else:
    #         a, v, out = model(spec, image)  # gray colored
    for step, batch_data in enumerate(dataloader):
        t_ids = batch_data['text_input_ids'].to(device)
        t_mask = batch_data['text_attention_mask'].to(device)
        c_ids = batch_data['code_input_ids'].to(device)
        c_mask = batch_data['code_attention_mask'].to(device)
        label = batch_data['label'].to(device)
        missing_mod = batch_data['missing_mode'].to(device)
        optimizer.zero_grad()
        
        # === 修改点 1：根据开关选择模型调用方式 ===
        if args.model_type != 'textcode':
            # 融合直接返回 5 个值 (t, c, out, out_t, out_c)
            t, c, out, out_t, out_c = model(t_ids, t_mask, c_ids, c_mask, missing_mod)
        else:
            # 基础模型只返回 3 个值 (t, c, out)
            t, c, out = model(t_ids, t_mask, c_ids, c_mask)

            # === 修改点 2：将原本繁琐的单模态计算逻辑移入 else 分支 ===
            if args.fusion_method == 'sum':
                out_c = (torch.mm(c, torch.transpose(model.fusion_module.fc_y.weight, 0, 1)) +
                         model.fusion_module.fc_y.bias)
                out_t = (torch.mm(t, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) +
                         model.fusion_module.fc_x.bias)
            elif args.fusion_method == 'concat':
                weight_size = model.fusion_module.fc_out.weight.size(1)
                out_c = (torch.mm(c, torch.transpose(model.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                         + model.fusion_module.fc_out.bias / 2)
                out_t = (torch.mm(t, torch.transpose(model.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                         + model.fusion_module.fc_out.bias / 2)
            elif args.fusion_method == 'film' or args.fusion_method == 'gated':
                out_c = out
                out_t = out

        text_sim = -EU_dist(t, text_proto)  # B x n_class
        code_sim = -EU_dist(c, code_proto)  # B x n_class

        if args.modulation == 'Proto' and args.modulation_starts <= epoch <= args.modulation_ends:

            score_t_p = sum([softmax(text_sim)[i][label[i]] for i in range(text_sim.size(0))])
            score_c_p = sum([softmax(code_sim)[i][label[i]] for i in range(code_sim.size(0))])
            ratio_t_p = score_t_p / score_c_p

            score_c = sum([softmax(out_c)[i][label[i]] for i in range(out_c.size(0))])
            score_t = sum([softmax(out_t)[i][label[i]] for i in range(out_t.size(0))])
            ratio_t = score_t / score_c

            loss_proto_t = criterion(text_sim, label)
            loss_proto_c = criterion(code_sim, label)

            if ratio_t_p > 1:
                beta = 0  # text coef
                lam = 1 * args.alpha  # code coef
            elif ratio_t_p < 1:
                beta = 1 * args.alpha
                lam = 0
            else:
                beta = 0
                lam = 0
            loss = criterion(out, label) + beta * loss_proto_t + lam * loss_proto_c
            loss_c = criterion(out_c, label)
            loss_t = criterion(out_t, label)
        else:
            loss = criterion(out, label)
            loss_proto_c = criterion(code_sim, label)
            loss_proto_t = criterion(text_sim, label)
            loss_c = criterion(out_c, label)
            loss_t = criterion(out_t, label)

            score_t_p = sum([softmax(text_sim)[i][label[i]] for i in range(text_sim.size(0))])
            score_c_p = sum([softmax(code_sim)[i][label[i]] for i in range(code_sim.size(0))])
            ratio_t_p = score_t_p / score_c_p
            score_c = sum([softmax(out_c)[i][label[i]] for i in range(out_c.size(0))])
            score_t = sum([softmax(out_t)[i][label[i]] for i in range(out_t.size(0))])
            ratio_t = score_t / score_c

        if args.fusion_method == 'sum' or args.fusion_method == 'concat':
            # grad_a = torch.Tensor([]).to(device)
            # grad_v = torch.Tensor([]).to(device)
            # grad_a_fusion = torch.Tensor([]).to(device)
            # grad_v_fusion = torch.Tensor([]).to(device)
            #
            # loss_c.backward(retain_graph=True)
            # if args.dataset != 'CGMNIST':
            #     for parms in model.visual_net.parameters():
            #         grad_v = torch.cat((grad_v, parms.grad.flatten()), 0)
            # else:
            #     for parms in model.colored_net.parameters():
            #         grad_v = torch.cat((grad_v, parms.grad.flatten()), 0)
            # optimizer.zero_grad()
            #
            # loss_t.backward(retain_graph=True)
            # if args.dataset != 'CGMNIST':
            #     for parms in model.audio_net.parameters():
            #         grad_a = torch.cat((grad_a, parms.grad.flatten()), 0)
            # else:
            #     for parms in model.gray_net.parameters():
            #         grad_a = torch.cat((grad_a, parms.grad.flatten()), 0)
            # optimizer.zero_grad()

            loss.backward()
            # if args.dataset != 'CGMNIST':
            #     for parms in model.audio_net.parameters():
            #         grad_a_fusion = torch.cat((grad_a_fusion, parms.grad.flatten()), 0)
            #     for parms in model.visual_net.parameters():
            #         grad_v_fusion = torch.cat((grad_v_fusion, parms.grad.flatten()), 0)
            # else:
            #     for parms in model.gray_net.parameters():
            #         grad_a_fusion = torch.cat((grad_a_fusion, parms.grad.flatten()), 0)
            #     for parms in model.colored_net.parameters():
            #         grad_v_fusion = torch.cat((grad_v_fusion, parms.grad.flatten()), 0)
            #
            # # calculate the angle  期望的方向和实际更新的方向的差值
            # _, t_angle = dot_product_angle_tensor(grad_a, grad_a_fusion)
            # _, c_angle = dot_product_angle_tensor(grad_v, grad_v_fusion)
            # _t_angle += t_angle
            # _c_angle += c_angle
            #
            # a_amp, a_f_amp, t_diff = grad_amplitude_diff(grad_a, grad_a_fusion)
            # v_amp, v_f_amp, c_diff = grad_amplitude_diff(grad_v, grad_v_fusion)
            # _t_diff += t_diff
            # _c_diff += c_diff

            # f_angle.write(str(ratio_t) +
            #               "\t" + str(ratio_t_p) +
            #               "\t" + str(t_angle) +
            #               "\t" + str(c_angle) +
            #               "\t" + str(a_amp) +
            #               "\t" + str(a_f_amp) +
            #               "\t" + str(t_diff) +
            #               "\t" + str(v_amp) +
            #               "\t" + str(v_f_amp) +
            #               "\t" + str(c_diff) +
            #               "\n")
            # f_angle.flush()

            # print('ratio: ', ratio_t, ratio_t_p, t_angle, c_angle)
        else:
            loss.backward()
            print('ratio: ', ratio_t, ratio_t_p)
            # f_angle.write(str(ratio_t) +
            #               "\t" + str(ratio_t_p) +
            #               "\n")
            # f_angle.flush()

            t_angle = 0
            c_angle = 0
            _t_angle += t_angle
            _c_angle += c_angle
        print('loss: ', loss.data, 'loss_p_c: ', loss_proto_c.data, 'loss_p_t: ', loss_proto_t.data,
              'loss_c: ', loss_c.data, 'loss_t: ', loss_t.data)

        optimizer.step()

        _loss += loss.item()
        _loss_t += loss_t.item()
        _loss_c += loss_c.item()
        _loss_p_t += loss_proto_t.item()
        _loss_p_c += loss_proto_c.item()
        # _ratio_t += ratio_t
        # _ratio_t_p += ratio_t_p
        _ratio_t += ratio_t.item() if isinstance(ratio_t, torch.Tensor) else ratio_t
        _ratio_t_p += ratio_t_p.item() if isinstance(ratio_t_p, torch.Tensor) else ratio_t_p

    if args.optimizer == 'SGD':
        scheduler.step()
    # f_angle.close()

    return _loss / len(dataloader), _loss_t / len(dataloader), _loss_c / len(dataloader), \
           _loss_p_t / len(dataloader), _loss_p_c / len(dataloader), \
           _t_angle / len(dataloader), _c_angle / len(dataloader), \
           _ratio_t / len(dataloader), _ratio_t_p / len(dataloader), _t_diff / len(dataloader), _c_diff / len(dataloader)


def valid(args, model, device, dataloader, text_proto, code_proto):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    elif args.dataset == 'CGMNIST':
        n_classes = 10
    elif args.dataset == 'TextCode':
        n_classes = 2  # 新增这行
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_t = [0.0 for _ in range(n_classes)]
        acc_c = [0.0 for _ in range(n_classes)]

        acc_t_p = [0.0 for _ in range(n_classes)]
        acc_c_p = [0.0 for _ in range(n_classes)]

        # === 新增：装所有的真实标签和预测标签 ===
        all_true_labels = []
        all_pred_labels = []

        # for step, (spec, image, label) in enumerate(dataloader):

        #     spec = spec.to(device)
        #     image = image.to(device)
        #     label = label.to(device)

        #     if args.dataset != 'CGMNIST':
        #         a, v, out = model(spec.unsqueeze(1).float(), image.float())
        #     else:
        #         a, v, out = model(spec, image)  # gray colored
        for step, batch_data in enumerate(dataloader):
            t_ids = batch_data['text_input_ids'].to(device)
            t_mask = batch_data['text_attention_mask'].to(device)
            c_ids = batch_data['code_input_ids'].to(device)
            c_mask = batch_data['code_attention_mask'].to(device)
            label = batch_data['label'].to(device)
            missing_mod = batch_data['missing_mode'].to(device)


            # === 根据开关选择模型调用方式 ===
            if args.model_type != 'textcode':
                # 直接返回 5 个值 (t, c, out, out_t, out_c)
                # missing_mod = torch.full((label.size(0),), 2).to(device)
                t, c, out, out_t, out_c = model(t_ids, t_mask, c_ids, c_mask, missing_mod)
            else:
                # 基础模型只返回 3 个值 (t, c, out)
                t, c, out = model(t_ids, t_mask, c_ids, c_mask)

                # === 修改点 2：将原本繁琐的单模态计算逻辑移入 else 分支 ===
                if args.fusion_method == 'sum':
                    out_c = (torch.mm(c, torch.transpose(model.fusion_module.fc_y.weight, 0, 1)) +
                            model.fusion_module.fc_y.bias)
                    out_t = (torch.mm(t, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) +
                            model.fusion_module.fc_x.bias)
                elif args.fusion_method == 'concat':
                    weight_size = model.fusion_module.fc_out.weight.size(1)
                    out_c = (torch.mm(c, torch.transpose(model.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                            + model.fusion_module.fc_out.bias / 2)
                    out_t = (torch.mm(t, torch.transpose(model.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                            + model.fusion_module.fc_out.bias / 2)
                elif args.fusion_method == 'film' or args.fusion_method == 'gated':
                    out_c = out
                    out_t = out

            prediction = softmax(out)
            pred_c = softmax(out_c)
            pred_t = softmax(out_t)

            text_sim = -EU_dist(t, text_proto)  # B x n_class
            code_sim = -EU_dist(c, code_proto)  # B x n_class
            # print(text_sim, code_sim, (text_sim != text_sim).any(), (code_sim != code_sim).any())
            pred_c_p = softmax(code_sim)
            pred_t_p = softmax(text_sim)

            for i in range(label.shape[0]):
                ma = np.argmax(prediction[i].cpu().data.numpy())
                c_pred = np.argmax(pred_c[i].cpu().data.numpy())
                t_pred = np.argmax(pred_t[i].cpu().data.numpy())
                c_p_pred = np.argmax(pred_c_p[i].cpu().data.numpy())
                t_p_pred = np.argmax(pred_t_p[i].cpu().data.numpy())
                num[label[i]] += 1.0

                # pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == c_pred:
                    acc_c[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == t_pred:
                    acc_t[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == c_p_pred:
                    acc_c_p[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == t_p_pred:
                    acc_t_p[label[i]] += 1.0
                # === 新增：保存每一道题的真实答案和模型的最终预测装 ===
                all_true_labels.append(label[i].cpu().item())
                all_pred_labels.append(ma)
    # === 新增：交卷后，让 sklearn 统一算分 (使用 macro 宏平均，对 0类和 1类一视同仁) ===
    # precision = precision_score(all_true_labels, all_pred_labels, average='macro', zero_division=0)
    # recall = recall_score(all_true_labels, all_pred_labels, average='macro', zero_division=0)
    # f1 = f1_score(all_true_labels, all_pred_labels, average='macro', zero_division=0)


    precision_none = precision_score(all_true_labels, all_pred_labels, average=None, zero_division=0)
    recall_none = recall_score(all_true_labels, all_pred_labels, average=None, zero_division=0)
    f1_none = f1_score(all_true_labels, all_pred_labels, average=None, zero_division=0)

    precision = precision_none.mean()
    recall = recall_none.mean()
    f1 = f1_none.mean()

    print(f"\n[类别 0] Precision: {precision_none[0]:.4f}, Recall: {recall_none[0]:.4f}, F1: {f1_none[0]:.4f}")
    print(f"[类别 1] Precision: {precision_none[1]:.4f}, Recall: {recall_none[1]:.4f}, F1: {f1_none[1]:.4f}")

    return sum(acc) / sum(num), sum(acc_t) / sum(num), sum(acc_c) / sum(num), \
           sum(acc_t_p) / sum(num), sum(acc_c_p) / sum(num), precision, recall, f1


def calculate_prototype(args, model, dataloader, device, epoch, t_proto=None, c_proto=None):
    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    elif args.dataset == 'CGMNIST':
        n_classes = 10
    elif args.dataset == 'TextCode':
        n_classes = 2
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    text_prototypes = torch.zeros(n_classes, args.embed_dim).to(device)
    code_prototypes = torch.zeros(n_classes, args.embed_dim).to(device)
    count_class = [0 for _ in range(n_classes)]

    # calculate prototype
    model.eval()
    with torch.no_grad():
        sample_count = 0
        all_num = len(dataloader)
        # for step, (spec, image, label) in enumerate(dataloader):
        #     spec = spec.to(device)  # B x 257 x 1004
        #     image = image.to(device)  # B x 3(image count) x 3 x 224 x 224
        #     label = label.to(device)  # B

        #     # TODO: make it simpler and easier to extend
        #     if args.dataset != 'CGMNIST':
        #         a, v, out = model(spec.unsqueeze(1).float(), image.float())
        #     else:
        #         a, v, out = model(spec, image)  # gray colored

        for step, batch_data in enumerate(dataloader):
            t_ids = batch_data['text_input_ids'].to(device)
            t_mask = batch_data['text_attention_mask'].to(device)
            c_ids = batch_data['code_input_ids'].to(device)
            c_mask = batch_data['code_attention_mask'].to(device)
            label = batch_data['label'].to(device)
            missing_mod = batch_data['missing_mode'].to(device)

            # 把文本和代码喂给模型
            if args.model_type != 'textcode':
                # 只需要特征 t 和 c，忽略后面三个返回值
                t, c, _, _, _ = model(t_ids, t_mask, c_ids, c_mask, missing_mod)
            else:
                t, c, out = model(t_ids, t_mask, c_ids, c_mask)

            for idx, l in enumerate(label):
                l = l.long()
                count_class[l] += 1
                text_prototypes[l, :] += t[idx, :]
                code_prototypes[l, :] += c[idx, :]

            sample_count += 1
            if args.dataset == 'AVE':
                pass
            else:
                if sample_count >= all_num // 10:
                    break
    for c in range(text_prototypes.shape[0]):
        text_prototypes[c, :] /= count_class[c]
        code_prototypes[c, :] /= count_class[c]

    if epoch <= 0:
        text_prototypes = text_prototypes
        code_prototypes = code_prototypes
    else:
        text_prototypes = (1 - args.momentum_coef) * text_prototypes + args.momentum_coef * t_proto
        code_prototypes = (1 - args.momentum_coef) * code_prototypes + args.momentum_coef * c_proto
    return text_prototypes, code_prototypes


def main():
    args = get_arguments()
    args.use_cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    setup_seed(args.random_seed)

    device = torch.device('cuda:' + str(args.gpu) if args.use_cuda else 'cpu')

    if args.dataset == 'TextCode':
        if args.model_type == 'prompt_textcode':
            print(">>> 使用 PromptTextCodeClassifier (PMR + MPLMM PromptModel) <<<")
            hyp_params = MPLMMConfig(args)
            model = PromptTextCodeClassifier(args, hyp_params, n_classes=2)
        elif args.model_type == 'mult_textcode':
            print(">>> 使用 MULTTextCodeClassifier (PMR + MPLMM MULTModel) <<<")
            hyp_params = MPLMMConfig(args)
            model = MULTTextCodeClassifier(args, hyp_params, n_classes=2)
        else:
            print(">>> 使用基础 TextCodeClassifier <<<")
            model = TextCodeClassifier(args, n_classes=2)
        print(">>> 不对 TextCode 模型进行权重初始化 (使用预训练 BERT/CodeBERT) <<<")
    else:
        raise NotImplementedError('Only TextCode dataset is supported. Got: {}'.format(args.dataset))
    model.to(device)

    # model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)
    elif args.optimizer == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
        scheduler = None
    elif args.optimizer == 'Adam':
        if args.dataset == 'TextCode' and args.model_type != 'textcode':
            print(f">>> 开启差异化学习率：主干(BERT)保持极小学习率，新生模块使用正常学习率 {args.learning_rate}")
            fusion_params = (model.prompt_model.parameters() if args.model_type == 'prompt_textcode'
                        else model.mult_model.parameters())
            optimizer_grouped_parameters = [
                {'params': model.text_net.parameters(), 'lr': 2e-5},
                {'params': model.code_net.parameters(), 'lr': 2e-5},
                {'params': model.text_proj.parameters(), 'lr': args.learning_rate},
                {'params': model.code_proj.parameters(), 'lr': args.learning_rate},
                {'params': fusion_params, 'lr': args.learning_rate},
                {'params': model.classifier_t.parameters(), 'lr': args.learning_rate},
                {'params': model.classifier_c.parameters(), 'lr': args.learning_rate}
            ]
            optimizer = optim.Adam(optimizer_grouped_parameters, betas=(0.9, 0.99))
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
        scheduler = None

    # if args.dataset == 'VGGSound':
    #     train_dataset = VGGSound(args, mode='train')
    #     test_dataset = VGGSound(args, mode='test')
    # elif args.dataset == 'KineticSound':
    #     train_dataset = AVDataset(args, mode='train')
    #     test_dataset = AVDataset(args, mode='test')
    # elif args.dataset == 'CREMAD':
    #     train_dataset = CramedDataset(args, mode='train')
    #     test_dataset = CramedDataset(args, mode='test')
    # elif args.dataset == 'AVE':
    #     train_dataset = AVEDataset(args, mode='train')
    #     test_dataset = AVEDataset(args, mode='test')
    #     val_dataset = AVEDataset(args, mode='val')
    # elif args.dataset == 'CGMNIST':
    #     train_dataset = CGMNISTDataset(args, mode='train')
    #     test_dataset = CGMNISTDataset(args, mode='test')
    #     val_dataset = CGMNISTDataset(args, mode='test')
    # else:
    #     raise NotImplementedError('Incorrect dataset name {}! '
    #                               'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

# === 开始：数据读取与划分 ===
    print("正在读取并划分数据...")
    # master_df = pd.read_csv("official.csv") 

    # train_df, temp_df = train_test_split(
    #     master_df, test_size=0.2, random_state=42, stratify=master_df['label']
    # )
    # val_df, test_df = train_test_split(
    #     temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
    # )

    train_df = pd.read_csv("official_train.csv")
    val_df = pd.read_csv("official_val.csv")
    test_df = pd.read_csv("official_test.csv")

    train_dataset = TextCodeDataset(
        train_df, 
        drop_rate=args.drop_rate, 
        full_data=args.full_data
    )
    
    # 验证集和测试集：强行写死 full_data=True
    val_dataset = TextCodeDataset(
        val_df, 
        drop_rate=args.drop_rate, 
        full_data=True 
    )
    
    test_dataset = TextCodeDataset(
        test_df, 
        drop_rate=args.drop_rate, 
        full_data=True 
    )
    
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # === 结束 ===


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4, pin_memory=False)

    if args.dataset == 'AVE':
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=4, pin_memory=False)
    elif args.dataset == 'CGMNIST':
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                    shuffle=True, pin_memory=False)

    if args.train:

        trainloss_file = args.logs_path + '/Method-CE-Proto-grad-amp' + '/train_loss-' + args.dataset + '-' + args.fusion_method + '-bsz' + \
                         str(args.batch_size) + '-lr' + str(args.learning_rate) \
                         + '-epoch' + str(args.epochs) + '-' + args.modulation + str(args.alpha) + \
                         '-mon' + str(args.momentum_coef) + '-' + str(args.num_frame) + '-end' + str(args.modulation_ends) \
                         + '-optim-' + args.optimizer + 'small_data.txt'
        if not os.path.exists(args.logs_path + '/Method-CE-Proto-grad-amp'):
            os.makedirs(args.logs_path + '/Method-CE-Proto-grad-amp')

        save_path = args.ckpt_path + '/Method-CE-Proto-grad-amp' + '/model-' + args.dataset + '-' + args.fusion_method + '-bsz' + \
                    str(args.batch_size) + '-lr' + str(args.learning_rate) \
                    + '-epoch' + str(args.epochs) + '-' + args.modulation + str(args.alpha) + \
                    '-mon' + str(args.momentum_coef) + '-' + str(args.num_frame) + '-end' + str(args.modulation_ends) \
                    + '-optim-' + args.optimizer + 'small_data'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if (os.path.isfile(trainloss_file)):
            os.remove(trainloss_file)  # 删掉已有同名文件
        f_trainloss = open(trainloss_file, 'a')

        # === 新增：写入表头 ===
        header = "Epoch\tTotal_Loss\tLoss_t_p\tLoss_c_p\tLoss_t\tLoss_c\tAcc_Fusion\tAcc_t_p\tAcc_c_p\tAcc_t\tAcc_c\tPrecision\tRecall\tF1_Score\tRatio_t_p\tRatio_t\n"
        f_trainloss.write(header)
        f_trainloss.flush()

        best_acc = 0.0

        epoch = 0

        if args.dataset == 'AVE':
            text_proto, code_proto = calculate_prototype(args, model, val_dataloader, device, epoch)
        elif args.dataset == 'CGMNIST':
            text_proto, code_proto = calculate_prototype(args, model, val_dataloader, device, epoch)
        else:
            text_proto, code_proto = calculate_prototype(args, model, train_dataloader, device, epoch)

        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))


            s_time = time.time()
            batch_loss, batch_loss_t, batch_loss_c, batch_loss_t_p, batch_loss_c_p, t_angle, c_angle, ratio_t, ratio_t_p, \
               t_diff, c_diff = train_epoch(args, epoch, model, device, train_dataloader, optimizer, scheduler,
                              text_proto, code_proto)

            if args.dataset == 'AVE':
                text_proto, code_proto = calculate_prototype(args, model, val_dataloader, device, epoch, text_proto, code_proto)
            elif args.dataset == 'CGMNIST':
                text_proto, code_proto = calculate_prototype(args, model, val_dataloader, device, epoch,
                                                                text_proto, code_proto)
            else:
                text_proto, code_proto = calculate_prototype(args, model, train_dataloader, device, epoch, text_proto, code_proto)
            e_time = time.time()
            print('per epoch time: ', e_time - s_time)
            # print('proto22', text_proto[22], code_proto[22])
            # acc, acc_t, acc_c, acc_t_p, acc_c_p = valid(args, model, device, test_dataloader, text_proto, code_proto)
            acc, acc_t, acc_c, acc_t_p, acc_c_p, precision, recall, f1 = valid(args, model, device, val_dataloader, text_proto, code_proto)
             # === 修改：控制台打印得更漂亮专业 ===
            print(f'Epoch {epoch} 考试成绩 ---> Acc: {acc*100:.2f}% | Precision: {precision*100:.2f}% | Recall: {recall*100:.2f}% | F1-Score: {f1*100:.2f}%')
            print('epoch: ', epoch, 'loss: ', batch_loss, batch_loss_t_p, batch_loss_c_p)
            print('epoch: ', epoch, 'acc: ', acc, 'acc_c_p: ', acc_c_p, 'acc_t_p: ', acc_t_p)
            f_trainloss.write(str(epoch) +
                              "\t" + str(batch_loss) +
                              "\t" + str(batch_loss_t_p) +
                              "\t" + str(batch_loss_c_p) +
                              "\t" + str(batch_loss_t) +
                              "\t" + str(batch_loss_c) +
                              "\t" + str(acc) +
                              "\t" + str(acc_t_p) +
                              "\t" + str(acc_c_p) +
                              "\t" + str(acc_t) +
                              "\t" + str(acc_c) +
                              "\t" + str(precision) +
                              "\t" + str(recall) +
                              "\t" + str(f1) +
                            #   "\t" + str(t_angle) +
                            #   "\t" + str(c_angle) +
                              "\t" + str(ratio_t_p) +
                              "\t" + str(ratio_t) +
                            #   "\t" + str(t_diff) +
                            #   "\t" + str(c_diff) +
                              "\n")
            f_trainloss.flush()

            if acc > best_acc or (epoch + 1) % 10 == 0:
                if acc > best_acc:
                    best_acc = float(acc)
            
                print('Saving model....')
                torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict() if scheduler is not None else None,
                        'text_proto': text_proto,
                        'code_proto': code_proto
                    },
                    os.path.join(save_path, 'epoch-{}.pt'.format(epoch))
                )
                print('Saved model!!!')
        f_trainloss.close()

    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        # modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        # fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        # assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        # assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        print("正在计算测试所需的特征原型...")
        epoch = 0 # 假定为第0轮，用于初始化
        # 使用验证集或测试集来计算原型
        text_proto, code_proto = calculate_prototype(args, model, val_dataloader, device, epoch)

        model.load_state_dict(state_dict)
        print('Trained model loaded!')

        # acc, acc_t, acc_c = valid(args, model, device, test_dataloader)
        acc, acc_t, acc_c, acc_t_p, acc_c_p, precision, recall, f1 = valid(args, model, device, test_dataloader, text_proto, code_proto)
        print('Accuracy: {}, accuracy_t: {}, accuracy_c: {}, precision: {}, recall: {}, f1: {}'.format(acc, acc_t, acc_c, precision, recall, f1))


if __name__ == "__main__":
    main()
