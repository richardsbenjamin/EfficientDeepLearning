from factorisation.densenet import get_student_densenet
from hyperparams.hyperparams_dict import hp_categorical
from utils import (
    get_best_transformations,
    get_device,
    get_test_cifar10_dataloader,
    get_train_cifar10_dataloader,
    load_trained_model,
    load_untrained_model,
    run_epochs,
)

if __name__ == "__main__":

    train_transforms = get_best_transformations()
    train_loader = get_train_cifar10_dataloader(transform=train_transforms)
    test_loader = get_test_cifar10_dataloader()

    teacher_model, _ = load_trained_model()

    growth_rates = [24, 32]

    for growth_rate in growth_rates:

        student_model = get_student_densenet(growth_rate=growth_rate)
        train_details = load_untrained_model(student_model)
        device = get_device()
        student_model.to(device)

        _, _, _ = run_epochs(
            train_details["model"],
            train_loader,
            test_loader,
            train_details,
            n_epochs=150,
            teacher=teacher_model,
            checkpoint_file_name=f"model_distill_train_student_size_{growth_rate}"
        )


