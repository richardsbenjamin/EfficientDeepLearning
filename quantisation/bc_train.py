from torch import nn

from quantisation.binaryconnect import BC
from utils import (
    get_best_transformations,
    get_device,
    get_cifar10_train_val_loaders,
    get_test_cifar10_dataloader,
    load_trained_model,
    load_untrained_model,
    pickle_dump,
    run_epochs,
)

if __name__ == "__main__":

    model, _ = load_trained_model()
    mymodelbc = BC(model)
    train_details = load_untrained_model("DenseNet121")
    device = get_device()

    transforms_ = get_best_transformations()
    train_loader, _ = get_cifar10_train_val_loaders(transform=transforms_)
    test_loader = get_test_cifar10_dataloader()

    n_epochs = 50

    best_test_acc, _, _ = run_epochs(
        mymodelbc,
        train_loader,
        test_loader,
        train_details,
        n_epochs,
        clip=True,
    )

    res = {
        "bc_test_acc": best_test_acc,
        "bc_model": mymodelbc,
    }
    pickle_dump(res, "bc_test_acc")


    

