import os
import pickle

import torch
import torch.optim as optim
import torchvision.transforms.v2 as transforms
from numpy import load, random
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10

from _typing_ import Dataset, Optimizer, LRScheduler, Transform


DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
ROOT_DIR = "/opt/img/effdl-cifar10/"
SPLITS_FILE = "cifar_train_val_splits.npz"


def pickle_dump(to_pickle, file_name: str):
    with open(f"{file_name}.pkl", "wb") as f:
        pickle.dump(to_pickle, f)

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_dataloader(dataset: Dataset, batchsize: int = 32) -> DataLoader:
    return DataLoader(dataset, batch_size=batchsize, shuffle=True)

def subset_data(dataset: Dataset, num_subset: int = 15000, seed: int = 2147483647):
    indices = list(range(len(dataset)))
    random.RandomState(seed=seed).shuffle(indices)
    return Subset(dataset, indices[:num_subset])

def get_train_cifar10_dataset(
        transform: Transform = DEFAULT_TRANSFORM,
        rootdir: str = ROOT_DIR,
        subset: bool = False,
        **subset_kwargs: dict,
    ) -> Dataset:
    dataset = CIFAR10(rootdir, train=True, download=True, transform=transform)
    if subset:
        dataset = subset_data(dataset, **subset_kwargs)
    return dataset

def get_cifar10_train_val_loaders(batchsize: int = 32, **subset_kwargs) -> tuple[DataLoader]:
    train_set, val_set = get_cifar10_train_val_subsets(**subset_kwargs)
    return (
        get_dataloader(train_set, batchsize),
        get_dataloader(val_set, batchsize),
    )

def get_cifar10_train_val_subsets(split_file: str = SPLITS_FILE, **train_kwargs: dict) -> tuple[Subset, Subset]:
    dataset = get_train_cifar10_dataset(**train_kwargs)
    split = load(split_file)
    train_idx = split["train"]
    val_idx = split["val"]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def get_train_cifar10_dataloader(transform: None = None, rootdir: str = ROOT_DIR, batchsize: int = 32, subset: bool = False) -> DataLoader:
    ...
    return get_dataloader(
        dataset,
        batchsize,
    )

def get_val_cifar10_dataloader() -> DataLoader:
    ...

def get_test_cifar10_dataloader(transform, rootdir: str = ROOT_DIR, batchsize: int = 32) -> DataLoader:
    return get_dataloader(
        CIFAR10(rootdir, train=False, download=True, transform=transform),
        batchsize,
    )

def get_optimiser(optimser_name: str, model_params, **optim_kwargs: dict) -> Optimizer:
    optim_class = getattr(optim, optimser_name)
    return optim_class(
        model_params,
        **optim_kwargs,
    )

def get_scheduler(scheduler_name: str, optimiser: Optimizer, **scheduler_params: dict) -> LRScheduler:
    scheduler_class = getattr(optim.lr_scheduler, scheduler_name)
    return scheduler_class(
        optimiser,
        **scheduler_params,
    )


def train(train_loader, net, optimiser, criterion, device: str = "cuda") -> tuple[float]:
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimiser.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimiser.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100.* correct / total

    return acc, train_loss

def test(test_loader, net, criterion, device: str = "cuda", half: bool = False) -> tuple[float]:
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if half:
                inputs = inputs.half()

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.* correct / total
    return acc, test_loss

def run_epochs(net, train_loader, test_loader, hyperparams: dict, start_epoch: int = 0, n_epochs: int = 200, device: str = "cuda"):
    best_acc = 0
    for epoch in range(start_epoch, start_epoch+n_epochs):

        print('Epoch:', epoch)

        train_acc, train_loss = train(
            train_loader,
            net,
            hyperparams["optimiser"],
            hyperparams["criterion"],
            device,
        )
        test_acc, test_loss = test(test_loader, net, hyperparams["criterion"], device)

        if test_acc > best_acc:
            save_checkpoint(net, test_acc, epoch)
            best_acc = test_acc

    return best_acc

def save_checkpoint(net, acc, epoch):
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('train_checkpoint'):
        os.mkdir('train_checkpoint')
    torch.save(state, './train_checkpoint/train_ckpt.pth')
