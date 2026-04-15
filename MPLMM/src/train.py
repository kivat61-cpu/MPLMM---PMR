import torch
from torch import nn
from src import model as mm
from src.utils import *
import torch.optim as optim
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau


from src.eval_metrics import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import warnings

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    # if hyp_params.pretrained_model is not None:
    #     model = getattr(mm, "PromptModel")(hyp_params)
    #     model = transfer_model(model, hyp_params.pretrained_model)
    # else:
    #     model = getattr(mm, "MULTModel")(hyp_params)
    model = getattr(mm, "PromptModel")(hyp_params)#为了使命令直接使用promptmodel

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=hyp_params.when, factor=0.1
    )
    settings = {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
    }
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings["model"]
    optimizer = settings["optimizer"]
    criterion = settings["criterion"]
    scheduler = settings["scheduler"]

    def train(model, optimizer, criterion):
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, (batch_X, batch_Y, missing_mod) in enumerate(train_loader):
            # text, audio, vision = batch_X
            # eval_attr = batch_Y.squeeze(-1)
            # model.zero_grad()

            # if hyp_params.use_cuda:
            #     with torch.cuda.device(0):
            #         text, audio, vision, eval_attr = (
            #             text.cuda(),
            #             audio.cuda(),
            #             vision.cuda(),
            #             eval_attr.cuda(),
            #         )
            #         if hyp_params.dataset == "iemocap":
            #             eval_attr = eval_attr.long()

            # batch_size = text.size(0)
            # net = nn.DataParallel(model) if batch_size > 10 else model
            # preds = net(text, audio, vision, missing_mod)

            text, code = batch_X
            eval_attr = batch_Y.squeeze(-1)
            model.zero_grad()

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    # 2. 放到 GPU 上的操作也要删掉 vision
                    text, code, eval_attr = (
                        text.cuda(),
                        code.cuda(),
                        eval_attr.cuda(),
                    )
                    if hyp_params.dataset == "iemocap":
                        eval_attr = eval_attr.long()

            batch_size = text.size(0)
            net = nn.DataParallel(model) if batch_size > 10 else model
            # 3. 输入模型的前向传播删掉第三个模态参数
            preds = net(text, code, missing_mod)

            preds = preds.squeeze(-1)

            if hyp_params.dataset == "iemocap":
                preds = preds.view(-1, 4)
                eval_attr = eval_attr.view(-1)
            
            if hyp_params.dataset == "csv":
                eval_attr = eval_attr.view(-1).long()
            
            raw_loss = criterion(preds, eval_attr)
            raw_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print(
                    "Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}".format(
                        epoch,
                        i_batch,
                        num_batches,
                        elapsed_time * 1000 / hyp_params.log_interval,
                        avg_loss,
                    )
                )
                proc_loss, proc_size = 0, 0
                start_time = time.time()

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
        results = []
        truths = []

        with torch.no_grad():
            # for i_batch, (batch_X, batch_Y, missing_mod) in enumerate(loader):
            #     text, audio, vision = batch_X
            #     eval_attr = batch_Y.squeeze(dim=-1)  # if num of labels is 1

            #     if hyp_params.use_cuda:
            #         with torch.cuda.device(0):
            #             text, audio, vision, eval_attr = (
            #                 text.cuda(),
            #                 audio.cuda(),
            #                 vision.cuda(),
            #                 eval_attr.cuda(),
            #             )
            #             if hyp_params.dataset == "iemocap":
            #                 eval_attr = eval_attr.long()

            #     batch_size = text.size(0)
            #     net = nn.DataParallel(model) if batch_size > 10 else model
            #     preds = net(text, audio, vision, missing_mod)

            for i_batch, (batch_X, batch_Y, missing_mod) in enumerate(loader):
                # 1. 解包
                text, code = batch_X
                eval_attr = batch_Y.squeeze(dim=-1)

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        # 2. 放到 GPU
                        text, code, eval_attr = (
                            text.cuda(),
                            code.cuda(),
                            eval_attr.cuda(),
                        )
                        if hyp_params.dataset == "iemocap":
                            eval_attr = eval_attr.long()

                batch_size = text.size(0)
                net = nn.DataParallel(model) if batch_size > 10 else model
                # 3. 输入模型
                preds = net(text, code, missing_mod)

                preds = preds.squeeze(-1)

                if hyp_params.dataset == "iemocap":
                    preds = preds.view(-1, 4)
                    eval_attr = eval_attr.view(-1)
                
                if hyp_params.dataset == "csv":
                    eval_attr = eval_attr.view(-1).long()

                total_loss += criterion(preds, eval_attr).item() * batch_size

                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train(model, optimizer, criterion)
        # val_loss, _, _ = evaluate(model, criterion, test=False)
        val_loss, val_results, val_truths = evaluate(model, criterion, test=False)
        test_loss, _, _ = evaluate(model, criterion, test=True)

        # ==================== 👇 我们新加的全天候监控仪表盘 👇 ====================
            
        # 忽略初始轮次可能因为瞎猜导致的“除以0”警告，保持日志清爽
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # 把张量转成数字数组
            preds_c = np.argmax(val_results.cpu().detach().numpy(), axis=1)
            truths_c = val_truths.cpu().detach().numpy().astype(int)
            
            
            e_acc = accuracy_score(truths_c, preds_c)
            e_f1 = f1_score(truths_c, preds_c, average='macro')
            e_prec = precision_score(truths_c, preds_c, average='macro')
            e_rec = recall_score(truths_c, preds_c, average='macro')
            
            # 打印
            print(f"    [Epoch {epoch:2d} 随堂测验] Acc: {e_acc:.4f} | F1: {e_f1:.4f} | Prec: {e_prec:.4f} | Rec: {e_rec:.4f}")
        # ==========================================================================

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)

        print("-" * 50)
        print(
            "Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}".format(
                epoch, duration, val_loss, test_loss
            )
        )
        print("-" * 50)

        if val_loss < best_valid:
            print(f"Saved model at {hyp_params.name}")
            torch.save(model, hyp_params.name)
            best_valid = val_loss

    model = torch.load(hyp_params.name, weights_only=False)
    _, results, truths = evaluate(model, criterion, test=False)

    if hyp_params.dataset == "mosei":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == "mosi":
        eval_mosi(results, truths, True)
    elif hyp_params.dataset == "iemocap":
        eval_iemocap(results, truths)
    elif hyp_params.dataset == "sims":
        eval_sims(results, truths)
    elif hyp_params.dataset == "csv":

        
        # 将 PyTorch 张量转为 NumPy 数组，并用 argmax 选出概率最大的分类 (0 或 1)
        preds_class = np.argmax(results.cpu().detach().numpy(), axis=1)
        truths_class = truths.cpu().detach().numpy().astype(int)
        
        acc = accuracy_score(truths_class, preds_class)
        f1 = f1_score(truths_class, preds_class, average='macro')
        
        print(f"\n========== 真正的二分类评估结果 ==========")
        print(f"Accuracy (准确率):  {acc:.4f}")
        print(f"F1 score: {f1:.4f}")
        print("真实标签前15个:", truths_class[:15])
        print("模型预测前15个:", preds_class[:15])
        print("==========================================\n")
    # =============== 新增这一块 csv 的专属算分逻辑 ===============
    # elif hyp_params.dataset == "csv":
    #     from sklearn.metrics import accuracy_score, f1_score, classification_report
    #     import numpy as np
    #     import warnings
        
    #     # 将 PyTorch 张量转为 NumPy 数组，并用 argmax 选出概率最大的分类 (0 ~ 20)
    #     preds_class = np.argmax(results.cpu().detach().numpy(), axis=1)
    #     truths_class = truths.cpu().detach().numpy().astype(int)
        
    #     acc = accuracy_score(truths_class, preds_class)
    #     # macro F1 能够一视同仁地看待大类和小类，对多分类很关键
    #     f1 = f1_score(truths_class, preds_class, average='macro') 
        
    #     print(f"\n========== 真正的多分类 (21类别) 评估结果 ==========")
    #     print(f"Overall Accuracy (整体准确率): {acc:.4f}")
    #     print(f"Macro F1 score (宏平均F1): {f1:.4f}")
    #     print("-" * 40)
    #     print("真实标签前15个:", truths_class[:15])
    #     print("模型预测前15个:", preds_class[:15])
    #     print("-" * 40)
        
    #     # 忽略由于某些类别在测试集中压根没出现而引发的除零警告
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         print("\n【各API类别详细体检报告】:")
    #         # 这行代码会打印出每一个类别的精确率、召回率和 F1 分数
    #         print(classification_report(truths_class, preds_class))
            
    #     print("==========================================\n")
    # =========================================================
