import optuna
import torch.nn as nn

import models
from _typing_ import DataLoader
from hyperparams.hyperparams_dict import hp_categorical
from factorisation.densenet import get_increasing_grouped_densenet121
from utils import (
    get_cifar10_train_val_loaders,
    get_device,
    get_optimiser,
    get_scheduler,
    pickle_dump,
    train, 
    test,
)

hp_floats = {
    "lr": {"low": 1e-4, "high": 0.1, "log": True},
    "momentum": {"low": 0.4, "high": 0.9},
    "weight_decay": {"low": 5e-4, "high": 1e-3},
}


def objective(
        trial,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 10,
        device: str = "cuda",
    ) -> float:
    algo_hyperparams = {
        hp_name: trial.suggest_float(hp_name, **hp_params)
        for hp_name, hp_params in hp_floats.items()
    }

    model = getattr(models, model_name)().to(device)

    del algo_hyperparams["momentum"]
    optimiser = get_optimiser(
        "AdamW", model.parameters(), **algo_hyperparams,
    )
    scheduler = get_scheduler(
        "CosineAnnealingLR",
        optimiser,
        **hp_categorical["lr_schedulers"]["params"]["CosineAnnealingLR"],
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(n_epochs):
        train_acc, train_loss = train(train_loader, model, optimiser, criterion, device)
        val_acc, val_loss = test(val_loader, model, criterion, device)

        scheduler.step()

        trial.report(val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_acc


if __name__ == "__main__":
    device = get_device()
    train_loader, val_loader = get_cifar10_train_val_loaders()

    # grouped1
    model_name = "model_grouped1"
    model = get_increasing_grouped_densenet121() 

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda x: objective(x, model, train_loader, val_loader, 10, device), n_trials=30)

    pickle_dump(study.best_trial, f"hp_best_trial_{model_name}")

    print(f"RUN SUCCESS {model}")

