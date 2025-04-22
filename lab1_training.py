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


def get_dataloader(dataset, batchsize: int = 32) -> DataLoader:
  return DataLoader(dataset, batch_size=batchsize, shuffle=True)

def train(train_loader, net, optimizer, criterion, epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
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

n_epochs = 200
start_epoch = 0



with open('hyperparameter_training_results.pkl', 'rb') as f:
    all_hyper_params = pickle.load(f)
    hyper_params = max(all_hyper_params, key=lambda x: x['acc'])

    lr = hyper_params["lr"]
    momentum = hyper_params["momentum"]
    weight_decay = hyper_params["weight_decay"]
    batch_size = hyper_params["batch_size"]


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

best_acc = 0
for epoch in range(start_epoch, start_epoch+n_epochs):

    print('Epoch:', epoch)

    train_loss = train(
    get_train_subset(batch_size),
    net,
    optimizer,
    criterion,
    epoch,
    )
    test_acc, test_loss = test(net, criterion, epoch)


    if test_acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('train_checkpoint'):
            os.mkdir('train_checkpoint')
        torch.save(state, './train_checkpoint/train_ckpt.pth')
        best_acc = test_acc


state = {
    'net': net.state_dict(),
    'model_name': 'VGG19',
    'hyper_params': hyper_params
}

torch.save(state, 'trained_model_state.pth')

print("SUCCESS RUN")