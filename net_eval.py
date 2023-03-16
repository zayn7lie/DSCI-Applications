from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

def eval(model, device, test_loader):
    model.eval()

    sum_o, sum_t = [], []
    for imgs, targets in test_loader:
        imgs, targets = imgs.to(device), targets.to(device)

        output, _ = model(imgs, None)
        output, targets = output.cpu().detach().numpy(), targets.cpu().detach().numpy()
        
        for i in output: sum_o.append(i)
        for i in targets: sum_t.append(i)
    sum_o, sum_t = np.transpose(sum_o), np.transpose(sum_t)

    f1, auc = 0, 0
    for i in range(8):
        temp_f1, temp_auc = f1_score(sum_t[i], np.array(sum_o[i] >= 0.5, dtype=float)) * 100, roc_auc_score(sum_t[i], sum_o[i]) * 100
        print("- {:.0f}th: F1: {:.4f}% Auc: {:.4f}%".format(i, temp_f1, temp_auc))
        f1 += temp_f1
        auc += temp_auc

    avgf1 = f1 / 8
    avgauc = auc / 8
    print("AVG: F1: {:.4f}% Auc: {:.4f}%\n".format(avgf1, avgauc))
