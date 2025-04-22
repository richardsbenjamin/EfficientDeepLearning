import numpy as np
from sklearn.model_selection import train_test_split

from utils import get_train_cifar10_dataset


dataset = get_train_cifar10_dataset()

targets = np.array(dataset.targets)
all_indices = np.arange(len(dataset))
train_idx, val_idx = train_test_split(all_indices, test_size=0.2, stratify=targets)

np.savez("cifar_train_val_splits.npz", train=train_idx, val=val_idx)

