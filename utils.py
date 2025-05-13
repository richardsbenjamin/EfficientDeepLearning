import os
import pickle

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import torchvision.transforms.v2 as transforms
from numpy import load, random
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10

import models
from hyperparams.hyperparams_dict import hp_categorical
from transforms.transformations_dict import transformations

from _typing_ import (
    Dataset,
    Optimizer,
    LRScheduler,
    Module,
    Transform,
)


DEFAULT_TRANSFORMS = [
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]
DEFAULT_TRANSFORM = transforms.Compose(DEFAULT_TRANSFORMS)
ROOT_DIR = "/opt/img/effdl-cifar10/"
SPLITS_FILE = "cifar_train_val_splits.npz"


def pickle_dump(to_pickle, file_name: str):
    with open(f"{file_name}.pkl", "wb") as f:
        pickle.dump(to_pickle, f)

def get_model_size_mb(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")/1e6
    os.remove('temp.p')
    return size

def get_hyperparams() -> dict:
    with open("hyperparams/hp_best_params.pkl", 'rb') as f:
        trial = pickle.load(f)
    return trial.params

def get_best_transformations() -> list:
    with open("transforms/transforms_best_trial.pkl", "rb") as f:
        res = pickle.load(f)

    transforms_list = []
    for var_name, bool_value in res["params"].items():
        if bool_value:
            transforms_list.append(
                transformations[var_name.replace("use", "")],
            )
    return transforms.Compose([
        *transforms_list,
        *DEFAULT_TRANSFORMS,
    ])

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

def get_train_cifar10_dataloader(transform: Transform = DEFAULT_TRANSFORM, rootdir: str = ROOT_DIR, batchsize: int = 32) -> DataLoader:
    return get_dataloader(
        CIFAR10(rootdir, train=True, download=True, transform=transform),
        batchsize,
    )

def get_test_cifar10_dataloader(transform: Transform = DEFAULT_TRANSFORM, rootdir: str = ROOT_DIR, batchsize: int = 32) -> DataLoader:
    return get_dataloader(
        CIFAR10(rootdir, train=False, download=True, transform=transform),
        batchsize,
    )

def load_untrained_model(model: str | Module) -> Module:
    """['model', 'scheduler', 'optimiser', 'criterion']"""
    hp_params = get_hyperparams()

    if isinstance(model, str):
        model_class = getattr(models, model)
        model = model_class()

    device = get_device()
    model.to(device)

    optimiser = get_optimiser(
        hp_params["optimizer"],
        model.parameters(),
        lr=hp_params["lr"],
        weight_decay=hp_params["weight_decay"]
    )

    s_name = hp_params["lr_scheduler"]
    scheduler = get_scheduler(
        s_name,
        optimiser,
        **hp_categorical["lr_schedulers"]["params"][s_name]
    )
    criterion = nn.CrossEntropyLoss()

    return {
        "model": model,
        "scheduler": scheduler,
        "optimiser": optimiser,
        "criterion": criterion,
    }

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

def train(
    train_loader: DataLoader,
    net: nn.Module,
    optimiser: Optimizer,
    criterion,
    device: str = "cuda",
    half: bool = False,
    clip: bool = False,
    mixup: bool = False,
) -> tuple[float, float]:
    net.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        if half:
            inputs = inputs.half()

        optimiser.zero_grad()

        if mixup:
            index = torch.randperm(inputs.size(0)).to(device)
            inputs_perm = inputs[index]
            targets_perm = targets[index]

            lam = random.random()
            inputs_mix = lam * inputs + (1 - lam) * inputs_perm
            outputs = net(inputs_mix)

            loss = lam * criterion(outputs, targets) + (1 - lam) * criterion(outputs, targets_perm)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        optimiser.step()

        if clip:
            net.clip()

        train_loss += loss.item()
        _, predicted = outputs.max(1)

        if mixup:
            total += 0
            correct += 0
        else:
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total if total > 0 else 0.0
    return acc, train_loss

def test(
        test_loader: DataLoader,
        net: nn.Module,
        criterion = nn.CrossEntropyLoss(),
        device: str = "cuda",
        half: bool = False,
    ) -> tuple[float]:
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

def run_epochs(
    net: Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    hyperparams: dict,
    n_epochs: int = 200,
    start_epoch: int = 0,
    device: str = "cuda",
    half: bool = False,
    clip: bool = False,
    mixup: bool = False,
):
    best_acc = 0
    train_accs = []
    test_accs = []

    for epoch in range(start_epoch, start_epoch + n_epochs):
        print(f"\nEpoch {epoch}")

        train_acc, train_loss = train(
            train_loader,
            net,
            hyperparams["optimiser"],
            hyperparams["criterion"],
            device,
            half=half,
            clip=clip,
            mixup=mixup
        )

        test_acc, test_loss = test(
            test_loader,
            net,
            hyperparams["criterion"],
            device,
            half
        )

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if test_acc > best_acc:
            save_checkpoint(net, test_acc, epoch)
            best_acc = test_acc

    return best_acc, train_accs, test_accs

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

def count_nonzero_parameters(model: nn.Module):
    return sum(torch.count_nonzero(p).item() for p in model.parameters())

def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def global_pruning(model: nn.Module, amount: float):
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

def combined_pruning(model: nn.Module, amount_structured: float, amount_unstructured: float):
    parameters_to_prune = []

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.ln_structured(module, name='weight', amount=amount_structured, n=2, dim=0)
            parameters_to_prune.append((module, 'weight')) 

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount_unstructured,
    )

    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')

def remove_pruning(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')

def run_global_pruning(model: nn.Module, testloader: "DataLoader", amount: float, half: bool = False) -> tuple[int, float]:
    global_pruning(model, amount=amount)
    acc, _ = test(testloader, model, half=half)
    remove_pruning(model)
    params = count_nonzero_parameters(model)
    return params, acc

def load_trained_model(device: str = "cuda") -> nn.Module:
    with open("train_results.pkl", "rb") as f:
        res = pickle.load(f)
    state_dict, acc, _, _ = res.values()

    model = models.DenseNet121()
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model, acc

def get_macs(model, input_size=(1, 3, 32, 32), device: str = "cuda", half: bool = False):
    macs = 0
    dummy_input = torch.randn(*input_size).to(device)

    def conv_hook(module, input, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            batch_size, out_channels, out_h, out_w = output.shape
            in_channels, _, kernel_h, kernel_w = module.weight.shape

            layer_macs = out_h * out_w * out_channels * kernel_h * kernel_w * in_channels
            macs += layer_macs

    def linear_hook(module, input, output):
        nonlocal macs
        if isinstance(module, nn.Linear):
            in_features, out_features = module.weight.shape
            layer_macs = in_features * out_features
            macs += layer_macs

    hooks = []
    for layer in model.children():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            hook = layer.register_forward_hook(conv_hook if isinstance(layer, nn.Conv2d) else linear_hook)
            hooks.append(hook)

    if half:
        dummy_input = dummy_input.half()
    model(dummy_input)

    for hook in hooks:
        hook.remove()

    return macs

def calculate_score(p_s, p_u, q_w, q_a, w, f, param_ref, ops_ref):
    param_score = ((1 - (p_s + p_u)) * (q_w / 32) * w) / param_ref
    ops_score = ((1 - p_s) * (max(q_w, q_a) / 32) * f) / ops_ref
    return param_score + ops_score

def zero_out_pruned_filters(model, prune_ratio):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):  
            weight = module.weight.data
            if weight is None:
                continue

            if hasattr(module, '_pruned_filters'):
                pruned_filters = module._pruned_filters
            else:
                pruned_filters = []

            remaining_filters = set(range(weight.shape[0])) - set(pruned_filters)
            if not remaining_filters:
                continue
            remaining_filters = list(remaining_filters)
            filter_norms = weight[remaining_filters].abs().sum(dim=(1, 2, 3) if isinstance(module, nn.Conv2d) else (1,))

            _, sorted_indices = torch.sort(filter_norms)

            num_filters_to_prune = int(prune_ratio * len(remaining_filters))

            filters_to_zero = sorted_indices[:num_filters_to_prune]
            filters_to_zero = [remaining_filters[idx] for idx in filters_to_zero]
            weight[filters_to_zero] = 0
            module._pruned_filters = pruned_filters + filters_to_zero
            
    return model