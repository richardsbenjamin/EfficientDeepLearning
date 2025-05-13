from hyperparams.hyperparams_dict import hp_categorical
from factorisation import densenet
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
    train_loader, _ = get_cifar10_train_val_loaders(transform=train_transforms)
    test_loader = get_test_cifar10_dataloader()

    get_functions = [
        "increasing_grouped_densenet121",
        "transition_grouped_densenet121",
        "increasing_transition_grouped_densenet121",
    ]

    n_epochs = 200

    for get_function_type in get_functions:

        model = getattr(densenet, f"get_{get_function_type}")()
        training_details = load_untrained_model(model)

        best_acc, train_accs, test_accs = run_epochs(
            model,
            train_loader,
            test_loader,
            training_details,
            n_epochs=n_epochs,
        )
        res = {
            'net': model.state_dict(),
            'acc': best_acc,
            'train_accs': train_accs,
            "val_accs": test_accs,
        }
        pickle_dump(res, f"train_results_{get_function_type}")

