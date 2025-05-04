from copy import deepcopy

import torch
from torch import nn

from utils import (
    calculate_score,
    count_nonzero_parameters,
    get_best_transformations,
    get_cifar10_train_val_loaders,
    get_device,
    get_macs,
    get_test_cifar10_dataloader,
    load_trained_model,
    load_untrained_model,
    pickle_dump,
    run_epochs,
    test,
)


def collect_outputs(model, layer, calibration_loader, device, half: bool = False):
    outputs = []

    def hook_fn(module, input, output):
        outputs.append(output.detach().cpu())

    handle = layer.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            inputs = inputs.to(device)
            if half:
                inputs = inputs.half()
            model(inputs)
            break
    handle.remove()
    return outputs[0]

def prune_filters_by_zeroing(layer, calibration_outputs, prune_ratio):
    num_filters = layer.weight.size(0)
    num_prune = int(num_filters * prune_ratio)
    importance = calibration_outputs.view(calibration_outputs.size(0), num_filters, -1).norm(2, dim=2).mean(0)
    _, prune_idx = torch.topk(importance, num_prune, largest=False)
    with torch.no_grad():
        layer.weight[prune_idx, :, :, :] = 0
        if layer.bias is not None:
            layer.bias[prune_idx] = 0
    return prune_idx



if __name__ == "__main__":
    train_transforms = get_best_transformations()
    train_loader, val_loader = get_cifar10_train_val_loaders(transform=train_transforms)
    test_loader = get_test_cifar10_dataloader()

    model, _ = load_trained_model()
    model.half()
    params_ref, ops_ref = count_nonzero_parameters(model), get_macs(model, half=True)
    train_details = load_untrained_model("DenseNet121")
    device = get_device()

    calibration_loader = train_loader
    prune_ratios = [0.2, 0.3, 0.4, 0.5]

    num_pruning_rounds = 3
    n_epochs = 20
    res = []

    for prune_ratio in prune_ratios:
        print("Prune ratio: ", prune_ratio)
        pruning_model = deepcopy(model)

        for pruning_round in range(num_pruning_rounds):
            print("Prune round: ", pruning_round)
            for layer in pruning_model.modules():
                if isinstance(layer, nn.Conv2d):
                    calibration_outputs = collect_outputs(
                        pruning_model, layer, calibration_loader, device, half=True,
                    )
                    prune_filters_by_zeroing(layer, calibration_outputs, prune_ratio)

            _, _, _ = run_epochs(
                model,
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
        params = count_nonzero_parameters(pruning_model)

        score = calculate_score(
            1 - (params / params_ref), 0, 16, 16, params, get_macs(pruning_model, half=True), params_ref, ops_ref
        )
        res.append(
            (prune_ratio, test_acc, score, params)
        )

    pickle_dump(res, "thinet_prune_half_quant")