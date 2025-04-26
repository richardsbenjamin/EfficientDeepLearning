import optuna
import torch.nn as nn
from torchvision.transforms import v2 as transformsv2

from hyperparams.hyperparams_explore import hp_categorical
from models import ResNet18, DenseNet121, PreActResNet18
from transforms.transformations_dict import transformations
from utils import (
    DEFAULT_TRANSFORMS,
    get_device,
    get_optimiser,
    get_scheduler,
    get_hyperparams,
    get_cifar10_train_val_loaders,
    pickle_dump,
    test,
    train,
)



def objective(
        trial,
        model_class: nn.Module,
        hp_params: dict,
        n_epochs: int = 20,
    ):
    # Create trial variables
    boolean_trials = {
        f"use_{key}": trial.suggest_categorical(f'use{key}', [True, False]) for key in transformations.keys()
    }

    # Create transform list
    transform_list = []
    
    for transform_name, transform in transformations.items():
        if boolean_trials[f"use_{transform_name}"]:
            transform_list.append(transform)

    transform_list.extend(DEFAULT_TRANSFORMS)
    transforms_ = transformsv2.Compose(transform_list)

    train_loader, val_loader = get_cifar10_train_val_loaders(transform=transforms_)
    
    model = model_class()
    device = get_device()
    model.to(device)
    
    optimiser = get_optimiser(
        hp_params["optimizer"],
        model.parameters(),
        lr=hp_params["lr"],
        weight_decay=hp_params["weight_decay"]
    )

    s_name = hp_params["lr_scheduler"]
    scheduler = get_scheduler(
        s_name,
        optimiser,
        **hp_categorical["lr_schedulers"]["params"][s_name]
    )

    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    for epoch in range(n_epochs):
        train_acc, train_loss = train(train_loader, model, optimiser, criterion, device)
        val_acc, val_loss = test(val_loader, model, criterion, device)

        if s_name == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        trial.report(val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_acc



if __name__ == "__main__":
    n_epochs = 20
    hp_params = get_hyperparams()
    chosen_model = DenseNet121

    study = optuna.create_study(direction="maximize") 
    study.optimize(
        lambda x: objective(x, chosen_model, hp_params, n_epochs),
        n_trials=15,
    )

    pickle_dump(study.best_trial, f"transforms_best_trial")

    print(f"RUN SUCCESS REGLAGE")






