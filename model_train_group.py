from hyperparams.hyperparams_dict import hp_categorical
from factorisation import model_functions
from utils import (
    get_test_cifar10_dataloader,
    get_train_cifar10_dataloader,
    load_untrained_model,
    run_epochs,
)

if __name__ == "__main__":

    train_loader = get_train_cifar10_dataloader()
    test_loader = get_test_cifar10_dataloader()

    n_epochs = 300

    grouped_type = "grouped2"

    model = model_functions[grouped_type]()
    train_details = load_untrained_model(model, f"hp_best_trial_{grouped_type}")

    _, _, _ = run_epochs(
        model,
        train_loader,
        test_loader,
        train_details,
        n_epochs=n_epochs,
        mixup=True,
        checkpoint_file_name=f"model_train_{grouped_type}",
    )

