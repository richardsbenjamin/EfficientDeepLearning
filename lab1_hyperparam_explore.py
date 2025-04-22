import os
import pickle

from torchvision.datasets import CIFAR10
import numpy as np
import torchvision.transforms as transforms
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


from models import VGG



## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network.
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

### The data from CIFAR10 are already downloaded in the following folder
rootdir = '/opt/img/effdl-cifar10/'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)

trainloader = DataLoader(c10train,batch_size=32,shuffle=True)
testloader = DataLoader(c10test,batch_size=32)



## number of target samples for the final dataset
num_train_examples = len(c10train)
num_samples_subset = 15000

## We set a seed manually so as to reproduce the results easily
seed  = 2147483647

## Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
indices = list(range(num_train_examples))
np.random.RandomState(seed=seed).shuffle(indices)## modifies the list in place

## We define the Subset using the generated indices
c10train_subset = torch.utils.data.Subset(c10train,indices[:num_samples_subset])
print(f"Initial CIFAR10 dataset has {len(c10train)} samples")
print(f"Subset of CIFAR10 dataset has {len(c10train_subset)} samples")


def get_train_subset(batchsize: int = 32) -> DataLoader:
  return DataLoader(c10train_subset,batch_size=batchsize,shuffle=True)

def train(trainloader_subset, net, optimizer, criterion, epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader_subset):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return train_loss


def test(net, criterion, epoch):
    global device
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    return acc, test_loss


device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = VGG('VGG19')
net = net.to(device)

hyperparameters = {
    "lr": [0.001, 0.01, 0.1],
    "momentum": [0.4, 0.6, 0.9],
    "weight_decay": [5e-4, 10e-4],
    "batchsize": [16, 32],
    "optimizer": ["SGD", "Adam", "RMSprop", "AdamW"],
    "lr_scheduler": ["StepLR", "ExponentialLR", "ReduceLROnPlateau", "CosineAnnealingLR"],s
}

hp_epochs = 10
start_epoch = 0

hp_res = []

for lr in hyperparameters["lr"]:
  for momentum in hyperparameters["momentum"]:
    for weight_decay in hyperparameters["weight_decay"]:
        for batch_size in hyperparameters["batchsize"]:
            try:
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(
                    net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
                
                best_acc = 0
                for epoch in range(start_epoch, start_epoch+hp_epochs):
                    print(f'Epoch: {epoch}')
                    print(f'Learning rate: {lr}')
                    print(f'Momentum: {momentum}')
                    print(f'Weight decay: {weight_decay}')
                    print(f'Batch size: {batch_size}')

                    train_loss = train(
                    get_train_subset(batch_size),
                    net,
                    optimizer,
                    criterion,
                    epoch,
                    )
                    test_acc, test_loss = test(net, criterion, epoch)

                    if test_acc > best_acc:
                        best_acc = test_acc

                hp_res.append({
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "acc": best_acc,
                })
            except torch.OutOfMemoryError as e:
                    print("Skipping config: Out of memory")
                    torch.cuda.empty_cache()  # Optional: clean up a bit



with open("hyperparameter_training_results.pkl", "wb") as f:
    pickle.dump(hp_res, f)

print("SUCCESS RUN")