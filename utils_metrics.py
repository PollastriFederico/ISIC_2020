from sklearn import metrics


def compute_accuracy_metrics(preds, gts):

    binary_preds = preds.argmax(dim=1, keepdim=False)

    tr_trues = float(((binary_preds * 2 - 1) == gts).sum().item())
    trues = float(binary_preds.sum().item())
    g_trues = float(gts.sum().item())
    corrects = float((binary_preds == gts).sum())
    acc = corrects / gts.size(0) * 100.
    pr = tr_trues / (trues + 10e-5)
    rec = tr_trues / (g_trues + 10e-5)
    fscore = (2 * pr * rec) / (pr + rec + 10e-5)
    auc = metrics.roc_auc_score(gts, preds[:, 1])

    return acc, pr, rec, fscore, auc
