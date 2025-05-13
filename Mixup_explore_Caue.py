import random
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Optimizer
from utils import (
    load_untrained_model,
    get_best_transformations,
    get_cifar10_train_val_loaders,
    get_test_cifar10_dataloader,
    run_epochs,
    pickle_dump
)
from _typing_ import (
    Dataset,
    Optimizer,
    LRScheduler,
    Module,
    Transform,
)

if __name__ == "__main__":
    model_dict = load_untrained_model("DenseNet121")
    model = model_dict["model"]

    train_transforms = get_best_transformations()
    train_loader, val_loader = get_cifar10_train_val_loaders(transform=train_transforms)
    test_loader = get_test_cifar10_dataloader()
    
    best_acc, train_accs, test_accs = run_epochs(
        model,
        train_loader,
        val_loader,
        model_dict,
        n_epochs=50,
        mixup=True
    )
    
    res = {
        'net': model.state_dict(),
        'acc': best_acc,
    }

    pickle_dump(res, "results_explore_mixup")