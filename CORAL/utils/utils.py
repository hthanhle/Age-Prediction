"""
UOW, Wed Feb 24 23:37:42 2021
Dependencies: torch>=1.1, torchvision>=0.3.0
"""
import numpy as np
import os
import torch
import torch.nn.functional as F


def get_ages(data_list_file):
    with open(data_list_file, 'r') as fd:
        imgs = fd.readlines()

    ages = []
    for k in range(len(imgs)):
        sample = imgs[k]
        splits = sample.split()
        age = np.int32(splits[1])
        ages.append(age)

    ages = np.array(ages)
    ages = torch.tensor(ages, dtype=torch.float)
    return ages


NUM_CLASSES = 117  # UTKFace: 0 - 116


def task_importance_weights(label_array):
    uniq = torch.unique(label_array)
    num_examples = label_array.size(0)

    m = torch.zeros(NUM_CLASSES)

    for i, t in enumerate(uniq):
        # print('t = ', t)
        m_k = torch.max(torch.tensor([label_array[label_array > t].size(0),
                                      num_examples - label_array[label_array > t].size(0)]))
        m[t.cpu().numpy()] = torch.sqrt(m_k.float())

    imp = m / torch.max(m)
    return imp


def loss_fn(logits, levels, imp):
    val = (-torch.sum((F.logsigmoid(logits) * levels
                       + (F.logsigmoid(logits) - logits) * (1 - levels)) * imp,
                      dim=1))
    return torch.mean(val)
