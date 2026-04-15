import torch
import pickle
import torch.nn.functional as F

# 1. 基础配置
model_path = 'finetuned_csv.pt' 
data_path = 'extracted_features.pkl'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 读取特征数据 (随便抽一题)
with open(data_path, 'rb') as f:
    data = pickle.load(f)

test_idx = 4  # 你可以随便换题号测试 (1, 2, 3...)
text_feat = torch.tensor(data['test']['text'][test_idx]).float().unsqueeze(0).to(device)
code_feat = torch.tensor(data['test']['audio'][test_idx]).float().unsqueeze(0).to(device)
true_label = int(data['test']['labels'][test_idx])

# 3. 唤醒模型
print("正在唤醒模型...")
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

# ================= 核心黑客技术：安装 Hook 监控 =================
# 我们要在模型最后一层分类器 (通常叫 out_layer) 之前，把特征截获下来
intercepted_features = {}

def hook_fn(module, input, output):
    # input[0] 就是进入最后一层分类器之前的“终极融合特征”
    intercepted_features['current'] = input[0].detach()

# 找到模型的最后一层并挂上钩子
# 注意：在 MULT/MPLMM 架构中，最后一层通常叫做 out_layer。
hook_handle = model.out_layer.register_forward_hook(hook_fn)
# ==============================================================

with torch.no_grad():
    print(f"\n========== 模态补全：内部特征相似度验证 ==========")
    
    # 【实验 1：获取完美特征 (传入真实 Text, 真实 Code)】
    X_full = (text_feat, code_feat)
    model(*X_full, torch.tensor([2]).to(device))
    perfect_vector = intercepted_features['current'].clone()
    print("✅ 成功截获【双模态完整】状态下的完美特征向量！")

    # 【实验 2：砍掉文本，获取脑补特征 (传入全 0 Text, 真实 Code)】
    X_only_code = (torch.zeros_like(text_feat), code_feat)
    model(*X_only_code, torch.tensor([0]).to(device))
    brain_completed_text_vector = intercepted_features['current'].clone()
    print("✅ 成功截获【缺文本，靠代码脑补】状态下的特征向量！")

    # 【实验 3：砍掉代码，获取脑补特征 (传入真实 Text, 全 0 Code)】
    X_only_text = (text_feat, torch.zeros_like(code_feat))
    model(*X_only_text, torch.tensor([1]).to(device))  # <-- 重点：这里正确使用了 X_only_text 和 [1]
    brain_completed_code_vector = intercepted_features['current'].clone()
    print("✅ 成功截获【缺代码，靠文本脑补】状态下的特征向量！")

    print("\n========== 开始计算余弦相似度 (Cosine Similarity) ==========")
    print("（注：相似度为 1.0 表示完全一模一样，0 表示毫无关联）\n")

    # 计算 缺文本补全 vs 完美特征 的相似度
    sim_text_missing = F.cosine_similarity(perfect_vector, brain_completed_text_vector, dim=1).item()
    print(f"👉 缺文本脑补特征 vs 完美特征 相似度: {sim_text_missing:.4f}")

    # 计算 缺代码补全 vs 完美特征 的相似度
    sim_code_missing = F.cosine_similarity(perfect_vector, brain_completed_code_vector, dim=1).item()
    print(f"👉 缺代码脑补特征 vs 完美特征 相似度: {sim_code_missing:.4f}")
    
    print("-" * 40)
    if sim_text_missing > 0.8 and sim_code_missing > 0.8:
        print("🎉 结论：极其强悍！模型内部的潜变量生成器完美补全了缺失的特征，成功骗过了分类器！")
    elif sim_text_missing > 0.5 and sim_code_missing > 0.5:
        print("👍 结论：补全机制生效了！虽然损失了一半数据，但模型依然努力复原了大部分语义特征！")
    else:
        print("⚠️ 结论：补全效果一般，可能你需要调整 --drop_rate 或者增加训练数据量。")

# 拆除钩子
hook_handle.remove()