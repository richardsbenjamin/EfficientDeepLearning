from copy import deepcopy

from utils import (
    count_nonzero_parameters,
    get_best_transformations,
    get_cifar10_train_val_loaders,
    get_test_cifar10_dataloader,
    load_trained_model,
    load_untrained_model,
    pickle_dump,
    run_epochs,
    test,
    zero_out_pruned_filters,
)




if __name__ == "___main__":
    acc = 100.
    n_epochs = 30
    prune_ratios = [0.2, 0.3, 0.4]

    train_transforms = get_best_transformations()
    train_loader, val_loader = get_cifar10_train_val_loaders(transform=train_transforms)
    test_loader = get_test_cifar10_dataloader()

    model, _ = load_trained_model()
    model.half()
    train_details = load_untrained_model("DenseNet121")

    res = {}
    for prune_ratio in prune_ratios:

        pruning_model = deepcopy(model)

        prune_track = []

        while acc > 85:
            pruning_model = zero_out_pruned_filters(pruning_model, prune_ratio=prune_ratio)
            params = count_nonzero_parameters(model)

            _, _, _ = run_epochs(
                pruning_model,
                train_loader,
                val_loader,
                train_details,
                n_epochs=n_epochs,
                half=True,
            )

            acc, _ = test(
                test_loader,
                pruning_model,
                half=True,
            )

            prune_track.append(
                (params, acc)
            )

        res[prune_ratio] = prune_track
        
    pickle_dump(res, "gradual_prune_half_quant_res")
    
    
    