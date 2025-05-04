from copy import deepcopy

from hyperparams.hyperparams_dict import hp_categorical
from utils import (
    calculate_score,
    count_nonzero_parameters,
    get_best_transformations,
    get_test_cifar10_dataloader,
    get_cifar10_train_val_loaders,
    get_macs,
    global_pruning,
    load_trained_model,
    load_untrained_model,
    pickle_dump,
    remove_pruning,
    test,
    run_epochs,
)

if __name__ == "__main__":

    train_transforms = get_best_transformations()
    train_loader, val_loader = get_cifar10_train_val_loaders(transform=train_transforms)
    test_loader = get_test_cifar10_dataloader()

    model, _ = load_trained_model()
    model.half()
    params_ref, ops_ref = count_nonzero_parameters(model), get_macs(model, half=True)
    train_details = load_untrained_model("DenseNet121")

    prune_amounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    n_epochs = 30

    retrain_res = []
    for amount in prune_amounts:
        print('AMOUNT: ', amount)
        pruning_model = deepcopy(model)

        global_pruning(pruning_model, amount=amount)

        _, _, _ = run_epochs(
            pruning_model,
            train_loader,
            val_loader,
            train_details,
            n_epochs=n_epochs,
            half=True,
        )
        test_acc, _ = test(
            test_loader,
            pruning_model,
            half=True,
        )

        remove_pruning(pruning_model)
        params = count_nonzero_parameters(pruning_model)

        score = calculate_score(
            0, 1 - (params / params_ref), 16, 16, params, get_macs(pruning_model), params_ref, ops_ref
        )
        retrain_res.append(
            (amount, test_acc, score)
        )

    pickle_dump(retrain_res, "global_retrain_results_half_quant")

