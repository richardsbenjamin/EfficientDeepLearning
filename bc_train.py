from torch import nn

from binaryconnect import BC
from utils import (
    get_best_transformations,
    get_device,
    get_cifar10_train_val_loaders,
    get_test_cifar10_dataloader,
    load_trained_model,
    load_untrained_model,
    pickle_dump,
    test,
)

if __name__ == "__main__":

    model, _ = load_trained_model()
    mymodelbc = BC(model)
    train_details = load_untrained_model("DenseNet121")
    device = get_device()

    transforms_ = get_best_transformations()
    train_loader, _ = get_cifar10_train_val_loaders(transform=transforms_)
    test_loader = get_test_cifar10_dataloader()

    best_acc = 0
    start_epoch = 0
    n_epochs = 20

    optimiser = train_details["optimiser"]
    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, start_epoch+n_epochs):
        print('EPOCH: ', epoch)
        mymodelbc.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimiser.zero_grad()
            outputs = mymodelbc(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimiser.step()
            mymodelbc.clip()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc, _ = test(
        test_loader,
        mymodelbc,
        criterion,
    )
    res = {
        "bc_test_acc": test_acc,
        "bc_model": mymodelbc,
    }
    pickle_dump(res, "bc_test_acc")


    

