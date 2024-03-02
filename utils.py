import torch


def one_hot_labels(labels, num_classes):
    output = torch.zeros((len(labels), num_classes))
    for i in range(len(labels)):
        output[i, labels[i]] = 1
    return output
