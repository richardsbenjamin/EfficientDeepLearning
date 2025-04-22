import optuna
import torch.nn as nn

import models
from _typing_ import DataLoader
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

hp_categorical = {
    "optimisers": {"names": ["SGD", "Adam", "RMSprop", "AdamW"]},
    "lr_schedulers": {
        "names": ["StepLR", "ExponentialLR", "ReduceLROnPlateau", "CosineAnnealingLR"],
        "params": {
            "StepLR": {"step_size": 5, "gamma": 0.5},
            "ExponentialLR": {"gamma": 0.9},
            "ReduceLROnPlateau": {"mode": 'min', "factor": 0.5, "patience": 3},
            "CosineAnnealingLR": {"T_max": 10},
        }
    },
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

    optimiser_name = trial.suggest_categorical("optimizer", hp_categorical["optimisers"]["names"])
    scheduler_name = trial.suggest_categorical("lr_scheduler", hp_categorical["lr_schedulers"]["names"])

    model = getattr(models, model_name)().to(device)

    if optimiser_name in ("Adam", "AdamW"):
        del algo_hyperparams["momentum"]

    optimiser = get_optimiser(
        optimiser_name, model.parameters(), **algo_hyperparams,
    )
    scheduler = get_scheduler(
        scheduler_name,
        optimiser,
        **hp_categorical["lr_schedulers"]["params"][scheduler_name],
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(n_epochs):
        train_acc, train_loss = train(train_loader, model, optimiser, criterion, device)
        val_acc, val_loss = test(val_loader, model, criterion, device)

        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        trial.report(val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_acc


models_ = [
    "ResNet18",
    "PreActResNet18",
    "DenseNet121",
]

device = get_device()
train_loader, val_loader = get_cifar10_train_val_loaders()

for model in models_:
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda x: objective(x, model, train_loader, val_loader, 20, device), n_trials=50)

    pickle_dump(study.best_trial, f"hp_best_trial_{model}")

    print(f"RUN SUCCESS {model}")