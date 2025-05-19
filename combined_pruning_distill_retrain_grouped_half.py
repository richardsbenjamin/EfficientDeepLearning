from copy import deepcopy

from hyperparams.hyperparams_dict import hp_categorical
from utils import (
    calculate_score,
    count_nonzero_parameters,
    get_train_cifar10_dataloader,
    get_test_cifar10_dataloader,
    get_macs,
    combined_pruning,
    load_trained_grouped_model,
    load_trained_model,
    load_untrained_model,
    pickle_dump,
    remove_pruning,
    test,
    run_epochs,
)

if __name__ == "__main__":

    group_type = "grouped1"

    train_loader = get_train_cifar10_dataloader()
    test_loader = get_test_cifar10_dataloader()

    teacher, _ = load_trained_model()
    teacher.half()

    PARAMS_RED, OPS_REF = count_nonzero_parameters(teacher), get_macs(teacher, half=True)

    student = load_trained_grouped_model(group_type)
    student.half()

    prune_amounts = [0.3, 0.6]
    n_epochs_list = [50, 150]

    retrain_res = []
    for i, amount in enumerate(prune_amounts):
        print('AMOUNT: ', amount)

        n_epochs = n_epochs_list[i]
        pruning_model = deepcopy(student)
        train_details = load_untrained_model(pruning_model, f"hp_best_trial_{group_type}")

        combined_pruning(pruning_model, amount_structured=amount, amount_unstructured=amount)

        _, _, _ = run_epochs(
            pruning_model,
            train_loader,
            test_loader,
            train_details,
            n_epochs=n_epochs,
            half=True,
            teacher=teacher,
            checkpoint_file_name=f"combined_pruning_distill_retrain_{group_type}_half_us_{amount}"
        )

