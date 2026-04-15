import torch
import pickle
import numpy as np

# 1. 基础配置
model_path = 'pretrained_csv.pt' # 模型名字
data_path = 'extracted_features.pkl'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 读取特征数据
with open(data_path, 'rb') as f:
    data = pickle.load(f)

# 3. 加载巅峰大脑
print("正在唤醒模型...")
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

# 初始化总计分板
total_samples = 0
correct_full = 0
correct_only_code = 0
correct_only_text = 0

print("\n========== 全量数据（122条）模态补全测试 ==========")

with torch.no_grad():
    # 依次遍历 训练集、验证集、测试集
    for split_name in ['train', 'valid', 'test']:
        split_size = len(data[split_name]['labels'])
        print(f"正在测试 {split_name} 集，共 {split_size} 条数据...")
        
        for i in range(split_size):
            total_samples += 1 # 总题数 +1
            
            # 提取当前题目的特征和正确答案
            text_feat = torch.tensor(data[split_name]['text'][i]).float().unsqueeze(0).to(device)
            code_feat = torch.tensor(data[split_name]['audio'][i]).float().unsqueeze(0).to(device)
            true_label = int(data[split_name]['labels'][i])

            # ================= 场景 A：完整输入 =================
            X_full = (text_feat, code_feat)
            missing_mod_full = torch.tensor([2]).to(device)
            pred_full = model(*X_full, missing_mod_full)
            if torch.argmax(pred_full, dim=1).item() == true_label:
                correct_full += 1

            # ================= 场景 B：只有代码，缺文本 =================
            X_only_code = (torch.zeros_like(text_feat), code_feat)
            missing_mod_text = torch.tensor([0]).to(device)
            pred_only_code = model(*X_only_code, missing_mod_text)
            if torch.argmax(pred_only_code, dim=1).item() == true_label:
                correct_only_code += 1

            # ================= 场景 C：只有文本，缺代码 =================
            X_only_text = (text_feat, torch.zeros_like(code_feat))
            missing_mod_code = torch.tensor([1]).to(device)
            pred_only_text = model(*X_only_text, missing_mod_code)
            if torch.argmax(pred_only_text, dim=1).item() == true_label:
                correct_only_text += 1

# 4. 结算最终成绩单 (计算准确率百分比)
acc_full = (correct_full / total_samples) * 100
acc_only_code = (correct_only_code / total_samples) * 100
acc_only_text = (correct_only_text / total_samples) * 100

print("\n========== 最终核心能力评估报告 ==========")
print(f"测试总数据量: {total_samples} 条")
print("-" * 45)
print(f"【场景 A: 双模态完整输入】 准确率: {acc_full:.2f}%  (答对 {correct_full}/{total_samples} 题)")
print(f"【场景 B: 仅给代码脑补文本】 准确率: {acc_only_code:.2f}%  (答对 {correct_only_code}/{total_samples} 题)")
print(f"【场景 C: 仅给文本脑补代码】 准确率: {acc_only_text:.2f}%  (答对 {correct_only_text}/{total_samples} 题)")
print("=========================================\n")