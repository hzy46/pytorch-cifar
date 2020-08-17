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
from utils import progress_bar
import time

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=128, type=int, help='total batch size')
parser.add_argument('--val_batch_size', default=128, type=int, help='val batch size')
parser.add_argument('--backend', choices=['gloo', 'nccl'], default='nccl', type=str)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--local_world_size", type=int, default=1)
args = parser.parse_args()


def setup():
    # initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )
    if args.backend == 'nccl':
        dist.init_process_group("nccl")
    elif args.backend == 'gloo':
        dist.init_process_group("gloo")
    return dist.get_world_size(), dist.get_rank()


def cleanup():
    dist.destroy_process_group()


def main(local_world_size, local_rank):
    # 假设每个worker只用一个gpu
    assert torch.cuda.device_count() == local_world_size
    world_size, rank = setup()
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
    # 训练分片应该按照world size和rank来切片
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
    # 放到local rank上
    net = net.to(local_rank)
    net = DDP(net, device_ids=[local_rank])

    if args.resume:
        # Load checkpoint.
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth', map_location=map_location)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)

    for epoch in range(start_epoch, start_epoch + 200):
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
            inputs, targets = inputs.to(local_rank), targets.to(local_rank)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

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

        # test
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(local_rank), targets.to(local_rank)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print('[%d][Epoch=%5d] Test Loss=%.3f Test Acc=%.3f%%' % (
                    rank,
                    epoch,
                    test_loss / (batch_idx + 1),
                    100. * correct / total,
                ))

        # Save checkpoint.
        if local_rank == 0:
            state = {
                'net': net.state_dict(),
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')

        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth', map_location=map_location)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']


if __name__ == '__main__':
    main(args.local_world_size, args.local_rank)