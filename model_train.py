from hyperparams.hyperparams_dict import hp_categorical
from utils import (
    get_best_transformations,
    get_test_cifar10_dataloader,
    get_cifar10_train_val_loaders,
    load_untrained_model,
    pickle_dump,
    test,
    run_epochs,
)

if __name__ == "__main__":

    train_transforms = get_best_transformations()
    train_loader, val_loader = get_cifar10_train_val_loaders(transform=train_transforms)
    test_loader = get_test_cifar10_dataloader()

    training_details = load_untrained_model("DenseNet121")

    best_acc, train_accs, test_accs = run_epochs(
        training_details["model"],
        train_loader,
        val_loader,
        training_details,
        n_epochs=500,
    )

    test_acc, test_loss = test(
        test_loader,
        training_details["model"],
    )

    res = {
        'net': training_details["model"].state_dict(),
        'acc': test_acc,
        'train_accs': train_accs,
        "val_accs": test_accs,
    }

    pickle_dump(res, "train_results")

