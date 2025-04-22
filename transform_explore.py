import optuna
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms import v2 as transforms

from utils import get_device, get_test_cifar10, get_train_cifar10, pickle_dump, run_epochs

from models import ResNet18, DenseNet, PreActResNet



rotation_degree = 45
translation = (0.2, 0.2) 
shearing = (10, 20, 0, 10)  # horizontal shear: 10–20°, vertical shear: 0–10°
kernel_size = 5

transformations = {
    # Geometric
    "rotation": transforms.RandomRotation(rotation_degree),
    "translation": transforms.RandomAffine(degrees=0, translate=translation),
    "shearing": transforms.RandomAffine(degrees=0, shear=shearing),

    # Non geometric
    "horizontal_flip": transforms.RandomHorizontalFlip(),
    "vertical_flip": transforms.RandomVerticalFlip(),
    "crop": transforms.RandomCrop(32, padding=4),
    "color_jitter": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    # "noise_injection": transforms.GaussianNoise(),
    "kernel": transforms.GaussianBlur(kernel_size=kernel_size),

    # Erasing
    "random_erasing": transforms.RandomErasing(),
}


def objective(trial):
    boolean_trials = {
        f"use_{key}": trial.suggest_categorical(f'use{key}', [True, False]) for key in transformations.keys()
    }
    transform_list = []
    
    for transform_name, transform in transformations.items():
        if boolean_trials[f"use_{transform_name}"]:
            transform_list.append(transform)

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform = transforms.Compose(transform_list)
    
    model = ResNet18()
    device = get_device()
    model.to(device)
    
    hyperparams = {
        "criterion": nn.CrossEntropyLoss(),
        "optimiser": optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    }
    subset = True
    
    return run_epochs(
        model,
        get_train_cifar10(transform=transform, subset=subset),
        get_test_cifar10(transform=transform),
        hyperparams,
        start_epoch=0,
        n_epochs=1,
    )


study = optuna.create_study(direction="maximize") 
study.optimize(objective, n_trials=10)

pickle_dump(
    study.best_trial.params,
    "best_transform",
)

print("RUN SUCCESS")

