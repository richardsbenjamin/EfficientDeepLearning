from factorisation import model_functions
from utils import (
    get_device,
    get_test_cifar10_dataloader,
    get_train_cifar10_dataloader,
    load_trained_model,
    load_untrained_model,
    run_epochs,
)

if __name__ == "__main__":
    train_loader = get_train_cifar10_dataloader()
    test_loader = get_test_cifar10_dataloader()

    teacher_model, _ = load_trained_model()

    grouped_types = ["grouped1", "grouped2"]


    for grouped_type in grouped_types:

        student_model = model_functions[grouped_type]()
        train_details = load_untrained_model(student_model, f"hp_best_trial_{grouped_type}")

        device = get_device()
        student_model.to(device)

        _, _, _ = run_epochs(
            train_details["model"],
            train_loader,
            test_loader,
            train_details,
            n_epochs=150,
            teacher=teacher_model,
            checkpoint_file_name=f"model_distill_train_{grouped_type}"
        )

