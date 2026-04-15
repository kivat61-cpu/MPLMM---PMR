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


from models.basic_model import AVClassifier, CGClassifier
from utils.utils import setup_seed, weight_init


import time


import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.TextCodeDataset import TextCodeDataset
from models.basic_model import TextCodeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score  # === 新增这行 ===
from models.basic_model import PromptTextCodeClassifier

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
    parser.add_argument('--use_mplmm', action='store_true', 
                        help='Turn on to use the fused PMR + MPLMM PromptModel')
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
                audio_proto, visual_proto, writer=None):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.train()
    print("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0
    _loss_p_a = 0
    _loss_p_v = 0

    _a_angle = 0
    _v_angle = 0
    _a_diff = 0
    _v_diff = 0
    _ratio_a = 0
    _ratio_a_p = 0

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
        optimizer.zero_grad()
        
        # === 修改点 1：根据开关选择模型调用方式 ===
        if args.use_mplmm:
            # 融合模型需要输入 missing_mod (2 代表训练时不缺失)
            # 并且直接返回 5 个值 (a, v, out, out_a, out_v)
            missing_mod = torch.full((label.size(0),), 2).to(device)
            a, v, out, out_a, out_v = model(t_ids, t_mask, c_ids, c_mask, missing_mod)
        else:
            # 基础模型只返回 3 个值 (a, v, out)
            a, v, out = model(t_ids, t_mask, c_ids, c_mask)

            # === 修改点 2：将原本繁琐的单模态计算逻辑移入 else 分支 ===
            if args.fusion_method == 'sum':
                out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_y.weight, 0, 1)) +
                         model.fusion_module.fc_y.bias)
                out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) +
                         model.fusion_module.fc_x.bias)
            elif args.fusion_method == 'concat':
                weight_size = model.fusion_module.fc_out.weight.size(1)
                out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                         + model.fusion_module.fc_out.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                         + model.fusion_module.fc_out.bias / 2)
            elif args.fusion_method == 'film' or args.fusion_method == 'gated':
                out_v = out
                out_a = out
        # 喂给模型
        # a, v, out = model(t_ids, t_mask, c_ids, c_mask)
        # if args.fusion_method == 'sum':
        #     out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_y.weight, 0, 1)) +
        #              model.fusion_module.fc_y.bias)
        #     out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) +
        #              model.fusion_module.fc_x.bias)
        # elif args.fusion_method == 'concat':
        #     weight_size = model.fusion_module.fc_out.weight.size(1)
        #     out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
        #              + model.fusion_module.fc_out.bias / 2)
        #     out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
        #              + model.fusion_module.fc_out.bias / 2)
        # elif args.fusion_method == 'film':
        #     out_v = out
        #     out_a = out
        # elif args.fusion_method == 'gated':
        #     out_v = out
        #     out_a = out

        audio_sim = -EU_dist(a, audio_proto)  # B x n_class
        visual_sim = -EU_dist(v, visual_proto)  # B x n_class
        # print('sim: ', audio_sim[0][0].data, visual_sim[0][0].data, a[0][0].data, v[0][0].data)

        if args.modulation == 'Proto' and args.modulation_starts <= epoch <= args.modulation_ends:

            score_a_p = sum([softmax(audio_sim)[i][label[i]] for i in range(audio_sim.size(0))])
            score_v_p = sum([softmax(visual_sim)[i][label[i]] for i in range(visual_sim.size(0))])
            ratio_a_p = score_a_p / score_v_p

            score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
            score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])
            ratio_a = score_a / score_v

            loss_proto_a = criterion(audio_sim, label)
            loss_proto_v = criterion(visual_sim, label)

            if ratio_a_p > 1:
                beta = 0  # audio coef
                lam = 1 * args.alpha  # visual coef
            elif ratio_a_p < 1:
                beta = 1 * args.alpha
                lam = 0
            else:
                beta = 0
                lam = 0
            loss = criterion(out, label) + beta * loss_proto_a + lam * loss_proto_v
            loss_v = criterion(out_v, label)
            loss_a = criterion(out_a, label)
        else:
            loss = criterion(out, label)
            loss_proto_v = criterion(visual_sim, label)
            loss_proto_a = criterion(audio_sim, label)
            loss_v = criterion(out_v, label)
            loss_a = criterion(out_a, label)

            score_a_p = sum([softmax(audio_sim)[i][label[i]] for i in range(audio_sim.size(0))])
            score_v_p = sum([softmax(visual_sim)[i][label[i]] for i in range(visual_sim.size(0))])
            ratio_a_p = score_a_p / score_v_p
            score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
            score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])
            ratio_a = score_a / score_v

        if args.fusion_method == 'sum' or args.fusion_method == 'concat':
            # grad_a = torch.Tensor([]).to(device)
            # grad_v = torch.Tensor([]).to(device)
            # grad_a_fusion = torch.Tensor([]).to(device)
            # grad_v_fusion = torch.Tensor([]).to(device)
            #
            # loss_v.backward(retain_graph=True)
            # if args.dataset != 'CGMNIST':
            #     for parms in model.visual_net.parameters():
            #         grad_v = torch.cat((grad_v, parms.grad.flatten()), 0)
            # else:
            #     for parms in model.colored_net.parameters():
            #         grad_v = torch.cat((grad_v, parms.grad.flatten()), 0)
            # optimizer.zero_grad()
            #
            # loss_a.backward(retain_graph=True)
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
            # _, a_angle = dot_product_angle_tensor(grad_a, grad_a_fusion)
            # _, v_angle = dot_product_angle_tensor(grad_v, grad_v_fusion)
            # _a_angle += a_angle
            # _v_angle += v_angle
            #
            # a_amp, a_f_amp, a_diff = grad_amplitude_diff(grad_a, grad_a_fusion)
            # v_amp, v_f_amp, v_diff = grad_amplitude_diff(grad_v, grad_v_fusion)
            # _a_diff += a_diff
            # _v_diff += v_diff

            # f_angle.write(str(ratio_a) +
            #               "\t" + str(ratio_a_p) +
            #               "\t" + str(a_angle) +
            #               "\t" + str(v_angle) +
            #               "\t" + str(a_amp) +
            #               "\t" + str(a_f_amp) +
            #               "\t" + str(a_diff) +
            #               "\t" + str(v_amp) +
            #               "\t" + str(v_f_amp) +
            #               "\t" + str(v_diff) +
            #               "\n")
            # f_angle.flush()

            # print('ratio: ', ratio_a, ratio_a_p, a_angle, v_angle)
        else:
            loss.backward()
            print('ratio: ', ratio_a, ratio_a_p)
            # f_angle.write(str(ratio_a) +
            #               "\t" + str(ratio_a_p) +
            #               "\n")
            # f_angle.flush()

            a_angle = 0
            v_angle = 0
            _a_angle += a_angle
            _v_angle += v_angle
        print('loss: ', loss.data, 'loss_p_v: ', loss_proto_v.data, 'loss_p_a: ', loss_proto_a.data,
              'loss_v: ', loss_v.data, 'loss_a: ', loss_a.data)

        optimizer.step()

        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()
        _loss_p_a += loss_proto_a.item()
        _loss_p_v += loss_proto_v.item()
        # _ratio_a += ratio_a
        # _ratio_a_p += ratio_a_p
        _ratio_a += ratio_a.item() if isinstance(ratio_a, torch.Tensor) else ratio_a
        _ratio_a_p += ratio_a_p.item() if isinstance(ratio_a_p, torch.Tensor) else ratio_a_p

    if args.optimizer == 'SGD':
        scheduler.step()
    # f_angle.close()

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), \
           _loss_p_a / len(dataloader), _loss_p_v / len(dataloader), \
           _a_angle / len(dataloader), _v_angle / len(dataloader), \
           _ratio_a / len(dataloader), _ratio_a_p / len(dataloader), _a_diff / len(dataloader), _v_diff / len(dataloader)


def valid(args, model, device, dataloader, audio_proto, visual_proto):
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
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        acc_a_p = [0.0 for _ in range(n_classes)]
        acc_v_p = [0.0 for _ in range(n_classes)]

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


            # === 根据开关选择模型调用方式 ===
            if args.use_mplmm:
                # 融合模型需要输入 missing_mod (2 代表训练时不缺失)
                # 并且直接返回 5 个值 (a, v, out, out_a, out_v)
                missing_mod = torch.full((label.size(0),), 2).to(device)
                a, v, out, out_a, out_v = model(t_ids, t_mask, c_ids, c_mask, missing_mod)
            else:
                # 基础模型只返回 3 个值 (a, v, out)
                a, v, out = model(t_ids, t_mask, c_ids, c_mask)

                # === 修改点 2：将原本繁琐的单模态计算逻辑移入 else 分支 ===
                if args.fusion_method == 'sum':
                    out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_y.weight, 0, 1)) +
                            model.fusion_module.fc_y.bias)
                    out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) +
                            model.fusion_module.fc_x.bias)
                elif args.fusion_method == 'concat':
                    weight_size = model.fusion_module.fc_out.weight.size(1)
                    out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                            + model.fusion_module.fc_out.bias / 2)
                    out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                            + model.fusion_module.fc_out.bias / 2)
                elif args.fusion_method == 'film' or args.fusion_method == 'gated':
                    out_v = out
                    out_a = out

            # a, v, out = model(t_ids, t_mask, c_ids, c_mask)
            # if args.fusion_method == 'sum':
            #     out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_y.weight, 0, 1)) +
            #              model.fusion_module.fc_y.bias)
            #     out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) +
            #              model.fusion_module.fc_x.bias)
            # elif args.fusion_method == 'concat':
            #     weight_size = model.fusion_module.fc_out.weight.size(1)
            #     out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
            #              + model.fusion_module.fc_out.bias / 2)
            #     out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
            #              + model.fusion_module.fc_out.bias / 2)
            # elif args.fusion_method == 'film':
            #     out_v = out
            #     out_a = out
            # elif args.fusion_method == 'gated':
            #     out_v = out
            #     out_a = out

            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            audio_sim = -EU_dist(a, audio_proto)  # B x n_class
            visual_sim = -EU_dist(v, visual_proto)  # B x n_class
            # print(audio_sim, visual_sim, (audio_sim != audio_sim).any(), (visual_sim != visual_sim).any())
            pred_v_p = softmax(visual_sim)
            pred_a_p = softmax(audio_sim)
            # print('pred_p: ', (pred_a_p != pred_a_p).any(), (pred_v_p != pred_v_p).any())

            # for i in range(image.shape[0]):
            for i in range(label.shape[0]):
                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                v_p = np.argmax(pred_v_p[i].cpu().data.numpy())
                a_p = np.argmax(pred_a_p[i].cpu().data.numpy())
                num[label[i]] += 1.0

                # pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v_p:
                    acc_v_p[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a_p:
                    acc_a_p[label[i]] += 1.0
                # === 新增：保存每一道题的真实答案和模型的最终预测装 ===
                all_true_labels.append(label[i].cpu().item())
                all_pred_labels.append(ma)
    # === 新增：交卷后，让 sklearn 统一算分 (使用 macro 宏平均，对 0类和 1类一视同仁) ===
    precision = precision_score(all_true_labels, all_pred_labels, average='macro', zero_division=0)
    recall = recall_score(all_true_labels, all_pred_labels, average='macro', zero_division=0)
    f1 = f1_score(all_true_labels, all_pred_labels, average='macro', zero_division=0)

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num), \
           sum(acc_a_p) / sum(num), sum(acc_v_p) / sum(num), precision, recall, f1


def calculate_prototype(args, model, dataloader, device, epoch, a_proto=None, v_proto=None):
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

    audio_prototypes = torch.zeros(n_classes, args.embed_dim).to(device)
    visual_prototypes = torch.zeros(n_classes, args.embed_dim).to(device)
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

            # 把文本和代码喂给模型，不要改接收变量的名字
            # a, v, out = model(t_ids, t_mask, c_ids, c_mask)
            if args.use_mplmm:
                missing_mod = torch.full((label.size(0),), 2).to(device)
                # 只需要特征 a 和 v，忽略后面三个返回值
                a, v, _, _, _ = model(t_ids, t_mask, c_ids, c_mask, missing_mod)
            else:
                a, v, out = model(t_ids, t_mask, c_ids, c_mask)

            for c, l in enumerate(label):
                l = l.long()
                count_class[l] += 1
                audio_prototypes[l, :] += a[c, :]
                visual_prototypes[l, :] += v[c, :]
                # if l == 22:
                #     print('fea_a', a[c, :], audio_prototypes[l, :])

            sample_count += 1
            if args.dataset == 'AVE':
                pass
            else:
                if sample_count >= all_num // 10:
                    break
    for c in range(audio_prototypes.shape[0]):
        audio_prototypes[c, :] /= count_class[c]
        visual_prototypes[c, :] /= count_class[c]

    if epoch <= 0:
        audio_prototypes = audio_prototypes
        visual_prototypes = visual_prototypes
    else:
        audio_prototypes = (1 - args.momentum_coef) * audio_prototypes + args.momentum_coef * a_proto
        visual_prototypes = (1 - args.momentum_coef) * visual_prototypes + args.momentum_coef * v_proto
    return audio_prototypes, visual_prototypes


def main():
    args = get_arguments()
    args.use_cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    setup_seed(args.random_seed)

    device = torch.device('cuda:' + str(args.gpu) if args.use_cuda else 'cpu')

    # if args.dataset == 'CGMNIST':
    #     model = CGClassifier(args)
    # else:
    #     model = AVClassifier(args)
    if args.dataset == 'CGMNIST':
        model = CGClassifier(args)
    elif args.dataset == 'TextCode':
        # model = TextCodeClassifier(args, n_classes=2)  # 文本-代码模型
        # 【修改】：根据开关选择实例化哪个模型
        if args.use_mplmm:
            print(">>> 使用融合模型 (PMR + MPLMM) <<<")
            # 定义 hyp_params
            hyp_params = MPLMMConfig(args)
            model = PromptTextCodeClassifier(args, hyp_params, n_classes=2)
        else:
            print(">>> 🟢 正在使用基础 PMR 模型 (TextCodeClassifier) <<<")
            model = TextCodeClassifier(args, n_classes=2)
    else:
        model = AVClassifier(args)

    if args.dataset != 'TextCode':
        model.apply(weight_init)
    else:
        print(">>> 不对 TextCodeClassifier 进行权重初始化 <<<")
    model.to(device)

    # model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)
    elif args.optimizer == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
        scheduler = None
    elif args.optimizer == 'Adam':
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
    master_df = pd.read_csv("official.csv") 

    train_df, temp_df = train_test_split(
        master_df, test_size=0.2, random_state=42, stratify=master_df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
    )

    train_dataset = TextCodeDataset(train_df)
    test_dataset = TextCodeDataset(test_df)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # === 结束 ===


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, pin_memory=False)

    if args.dataset == 'AVE':
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                    shuffle=False, pin_memory=False)
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
        header = "Epoch\tTotal_Loss\tLoss_a_p\tLoss_v_p\tLoss_a\tLoss_v\tAcc_Fusion\tAcc_a_p\tAcc_v_p\tAcc_a\tAcc_v\tPrecision\tRecall\tF1_Score\tRatio_a_p\tRatio_a\n"
        f_trainloss.write(header)
        f_trainloss.flush()

        best_acc = 0.0

        epoch = 0

        if args.dataset == 'AVE':
            audio_proto, visual_proto = calculate_prototype(args, model, val_dataloader, device, epoch)
        elif args.dataset == 'CGMNIST':
            audio_proto, visual_proto = calculate_prototype(args, model, val_dataloader, device, epoch)
        else:
            audio_proto, visual_proto = calculate_prototype(args, model, train_dataloader, device, epoch)

        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))


            s_time = time.time()
            batch_loss, batch_loss_a, batch_loss_v, batch_loss_a_p, batch_loss_v_p, a_angle, v_angle, ratio_a, ratio_a_p, \
               a_diff, v_diff = train_epoch(args, epoch, model, device, train_dataloader, optimizer, scheduler,
                              audio_proto, visual_proto)

            if args.dataset == 'AVE':
                audio_proto, visual_proto = calculate_prototype(args, model, val_dataloader, device, epoch, audio_proto, visual_proto)
            elif args.dataset == 'CGMNIST':
                audio_proto, visual_proto = calculate_prototype(args, model, val_dataloader, device, epoch,
                                                                audio_proto, visual_proto)
            else:
                audio_proto, visual_proto = calculate_prototype(args, model, train_dataloader, device, epoch, audio_proto, visual_proto)
            e_time = time.time()
            print('per epoch time: ', e_time - s_time)
            # print('proto22', audio_proto[22], visual_proto[22])
            # acc, acc_a, acc_v, acc_a_p, acc_v_p = valid(args, model, device, test_dataloader, audio_proto, visual_proto)
            acc, acc_a, acc_v, acc_a_p, acc_v_p, precision, recall, f1 = valid(args, model, device, test_dataloader, audio_proto, visual_proto)
             # === 修改：控制台打印得更漂亮专业 ===
            print(f'Epoch {epoch} 考试成绩 ---> Acc: {acc*100:.2f}% | Precision: {precision*100:.2f}% | Recall: {recall*100:.2f}% | F1-Score: {f1*100:.2f}%')
            print('epoch: ', epoch, 'loss: ', batch_loss, batch_loss_a_p, batch_loss_v_p)
            print('epoch: ', epoch, 'acc: ', acc, 'acc_v_p: ', acc_v_p, 'acc_a_p: ', acc_a_p)
            f_trainloss.write(str(epoch) +
                              "\t" + str(batch_loss) +
                              "\t" + str(batch_loss_a_p) +
                              "\t" + str(batch_loss_v_p) +
                              "\t" + str(batch_loss_a) +
                              "\t" + str(batch_loss_v) +
                              "\t" + str(acc) +
                              "\t" + str(acc_a_p) +
                              "\t" + str(acc_v_p) +
                              "\t" + str(acc_a) +
                              "\t" + str(acc_v) +
                              "\t" + str(precision) +
                              "\t" + str(recall) +
                              "\t" + str(f1) +
                            #   "\t" + str(a_angle) +
                            #   "\t" + str(v_angle) +
                              "\t" + str(ratio_a_p) +
                              "\t" + str(ratio_a) +
                            #   "\t" + str(a_diff) +
                            #   "\t" + str(v_diff) +
                              "\n")
            f_trainloss.flush()

            # if acc > best_acc or (epoch + 1) % 10 == 0:
            #     if acc > best_acc:
            #         best_acc = float(acc)
            
            #     print('Saving model....')
            #     torch.save(
            #         {
            #             'model': model.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #             'scheduler': scheduler.state_dict() if scheduler is not None else None
            #         },
            #         os.path.join(save_path, 'epoch-{}.pt'.format(epoch))
            #     )
            #     print('Saved model!!!')
        f_trainloss.close()

    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        model.load_state_dict(state_dict)
        print('Trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == "__main__":
    main()
