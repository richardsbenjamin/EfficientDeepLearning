
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, Compose
from factorisation import model_functions
from utils import (
    DEFAULT_TRANSFORM,
    DEFAULT_TRANSFORMS,
    get_cifar10_train_val_loaders,
    load_untrained_model,
    run_epochs,
)

autoaugment_transform = Compose([
    AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
    *DEFAULT_TRANSFORMS,
])

if __name__ == "__main__":

    n_epochs = 20
    grouped_type = "grouped1"

    model = model_functions[grouped_type]()
    train_details = load_untrained_model(model, f"hp_best_trial_{grouped_type}")

    transform_options = [
        {"autoaugment": True, "mixup": False},
        {"autoaugment": False, "mixup": True},
        {"autoaugment": True, "mixup": True},
    ]

    for transform_option in transform_options:
        print(transform_option)
        auto = transform_option["autoaugment"]
        mixup = transform_option["mixup"]
        train_loader, val_loader = get_cifar10_train_val_loaders(
            transform=autoaugment_transform if auto else DEFAULT_TRANSFORM,
        )
        _, _, _ = run_epochs(
            train_details["model"],
            train_loader,
            val_loader,
            train_details,
            n_epochs=n_epochs,
            mixup=mixup,
            checkpoint_file_name=f"train_transform_explore_{grouped_type}_auto_{auto}_mixup_{mixup}"
        )

    print("RUN SUCESS AUGO AUGMENT")