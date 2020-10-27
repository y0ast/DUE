import numpy as np
import torch
import gpytorch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from lib.datasets import get_dataset


def prepare_ood_datasets(true_dataset, ood_dataset):
    ood_dataset.transform = true_dataset.transform

    datasets = [true_dataset, ood_dataset]

    anomaly_targets = torch.cat(
        (torch.zeros(len(true_dataset)), torch.ones(len(ood_dataset)))
    )

    concat_datasets = torch.utils.data.ConcatDataset(datasets)

    dataloader = torch.utils.data.DataLoader(
        concat_datasets, batch_size=512, shuffle=False, num_workers=4, pin_memory=True
    )

    return dataloader, anomaly_targets


def loop_over_dataloader(model, likelihood, dataloader):
    model.eval()

    with torch.no_grad():
        scores = []
        accuracies = []
        for data, target in dataloader:
            data = data.cuda()
            target = target.cuda()

            if likelihood is None:
                output = torch.stack([m(data).exp() for m in model]).mean(0)
            else:
                with gpytorch.settings.num_likelihood_samples(32):
                    y_pred = model(data).to_data_independent_dist()
                    output = likelihood(y_pred).probs.mean(0)

            uncertainty = -(output * output.log()).sum(1)

            pred = torch.argmax(output, dim=1)
            accuracy = pred.eq(target)

            accuracies.append(accuracy.cpu().numpy())
            scores.append(uncertainty.cpu().numpy())

    scores = np.concatenate(scores)
    accuracies = np.concatenate(accuracies)

    return scores, accuracies


def get_ood_metrics(in_dataset, out_dataset, model, likelihood=None):
    _, _, _, in_dataset = get_dataset(in_dataset)
    _, _, _, out_dataset = get_dataset(out_dataset)

    dataloader, anomaly_targets = prepare_ood_datasets(in_dataset, out_dataset)

    scores, accuracies = loop_over_dataloader(model, likelihood, dataloader)

    accuracy = np.mean(accuracies[: len(in_dataset)])
    auroc = roc_auc_score(anomaly_targets, scores)

    precision, recall, _ = precision_recall_curve(anomaly_targets, scores)
    aupr = auc(recall, precision)

    return accuracy, auroc, aupr


def get_auroc_classification(data, model, likelihood=None):
    if isinstance(data, torch.utils.data.Dataset):
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=512, shuffle=False, num_workers=4, pin_memory=True
        )
    else:
        dataloader = data

    scores, accuracies = loop_over_dataloader(model, likelihood, dataloader)

    accuracy = np.mean(accuracies)
    roc_auc = roc_auc_score(1 - accuracies, scores)

    return accuracy, roc_auc
