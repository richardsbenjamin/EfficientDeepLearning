from copy import deepcopy

import torch

from factorisation import model_functions
from utils import (
    get_device,
    get_train_cifar10_dataloader,
    get_test_cifar10_dataloader,
    combined_pruning,
    load_trained_model,
    load_untrained_model,
    run_epochs,
)

if __name__ == "__main__":

    group_type = "grouped1"

    train_loader = get_train_cifar10_dataloader()
    test_loader = get_test_cifar10_dataloader()
    device = get_device()

    teacher, _ = load_trained_model()
    teacher.half()

    res = torch.load(f"train_checkpoint/model_distill_train_{group_type}.pth")
    model_distill_g1 = model_functions["grouped1"]()
    model_distill_g1.load_state_dict(res["net"])
    model_distill_g1.to(device)


    res = torch.load(f"train_checkpoint/model_train_{group_type}.pth")
    model_grouped1 = model_functions["grouped1"]()
    model_grouped1.load_state_dict(res["net"])
    model_grouped1.to(device)


    models = [model_distill_g1, model_grouped1]

    student = models[0]

    prune_amounts = [0.3, 0.6]
    n_epochs_list = [10, 10]

    retrain_res = []
    for i, amount in enumerate(prune_amounts):
        print('AMOUNT: ', amount)

        n_epochs = n_epochs_list[i]
        pruning_model = deepcopy(student)
        pruning_model.half()
        train_details = load_untrained_model(pruning_model, f"hp_best_trial_{group_type}")

        combined_pruning(pruning_model, amount_structured=0, amount_unstructured=amount)

        _, _, _ = run_epochs(
            pruning_model,
            train_loader,
            test_loader,
            train_details,
            n_epochs=n_epochs,
            half=True,
            teacher=teacher,
            checkpoint_file_name=f"unstruc_pruning_{amount}_distill_retrain_{group_type}_half"
        )

