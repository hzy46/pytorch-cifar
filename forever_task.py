'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
import time

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=128, type=int, help='total batch size')
parser.add_argument('--val_batch_size', default=128, type=int, help='val batch size')
parser.add_argument('--world_size', default=8, type=int)
parser.add_argument('--sleep_time_ms_every_step', default=100, type=int)
parser.add_argument('--backend', choices=['gloo', 'nccl'], default='nccl', type=str)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()


def setup(rank, world_size):
    # initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if args.backend == 'nccl':
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    elif args.backend == 'gloo':
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, sleep_time_ms_every_step):
    setup(rank, world_size)
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    # sampler 进行shuffle，参考https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    # 每个process应该是不同的数据切片
    # 对于DP数据切片是自动的，DDP要自己手动做数据切片
    trainsampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False, num_workers=2, sampler=trainsampler)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.val_batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    net = RegNetX_200MF()
    net = net.to(rank)
    net = DDP(net, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)

    epoch = start_epoch
    while True:
        # train
        print('\n[%d]Epoch: %d' % (rank, epoch))
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
        trainsampler.set_epoch(epoch)
        start_ts = time.time()
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(rank), targets.to(rank)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            time.sleep(sleep_time_ms_every_step / 1000.)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 5 == 0:
                print('[%d][Epoch=%5d][Step=%5d/%5d] Train Loss=%.3f Train Acc=%.3f%%' % (
                    rank,
                    epoch,
                    batch_idx + 1,
                    len(trainloader),
                    train_loss / (batch_idx+1),
                    100. * correct / total,
                ))
        print('[%d] Epoch %d Elapsed Time: %5ds' % (rank, epoch, int(time.time() - start_ts)))

        dist.barrier()
        epoch += 1


if __name__ == '__main__':
    mp.spawn(main,
             args=(args.world_size, args.sleep_time_ms_every_step),
             nprocs=args.world_size,
             join=True)