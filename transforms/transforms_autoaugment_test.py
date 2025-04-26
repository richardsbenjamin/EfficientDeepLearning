
import torch.nn as nn
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, Compose

from hyperparams.hyperparams_explore import hp_categorical
from models import DenseNet121
from utils import (
    DEFAULT_TRANSFORMS,
    get_cifar10_train_val_loaders,
    get_device, get_hyperparams,
    get_optimiser, get_scheduler,
    pickle_dump,
    test,
    train,
)


if __name__ == "__main__":

    n_epochs = 20
    hp_params = get_hyperparams()
    chosen_model = DenseNet121

    autoaugment_transform = Compose([
        AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        *DEFAULT_TRANSFORMS,
    ])
    train_loader, val_loader = get_cifar10_train_val_loaders(transform=autoaugment_transform)

    model = chosen_model()
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
        print('EPOCH', epoch)
        train_acc, train_loss = train(train_loader, model, optimiser, criterion, device)
        val_acc, val_loss = test(val_loader, model, criterion, device)

        if s_name == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc


    auto_augment_acc = {"autoaugment_acc": best_val_acc}
    pickle_dump(auto_augment_acc, "autoaugment_acc")

    print("RUN SUCESS AUGO AUGMENT")