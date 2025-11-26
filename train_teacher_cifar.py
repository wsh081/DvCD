import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np

import models
import torchvision
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds

from bisect import bisect_right
import time
import math

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--data', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset name')
parser.add_argument('--arch', default='wrn_40_2_distill', type=str, help='network architecture')  # 改为蒸馏模型
parser.add_argument('--init-lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--milestones', type=int, nargs='+', default=[150, 180, 210], help='milestones for lr-multistep')
parser.add_argument('--warmup-epoch', default=0, type=int, help='warmup epoch')
parser.add_argument('--epochs', type=int, default=240, help='number of epochs')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='the number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--kd_T', type=float, default=3, help='temperature for KD distillation')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--freezed', action='store_true', help='freezing backbone')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='directory for storing checkpoint files')
parser.add_argument('--pretrained-backbone', default='./pretrained_models/wrn_40_2.pth', type=str,
                    help='pretrained weights of backbone')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

log_txt = 'result/' + str(os.path.basename(__file__).split('.')[0]) + '_' + \
          'arch' + '_' + args.arch + '_' + \
          'dataset' + '_' + args.dataset + '_' + \
          'seed' + str(args.manual_seed) + '.txt'

log_dir = str(os.path.basename(__file__).split('.')[0]) + '_' + \
          'arch' + '_' + args.arch + '_' + \
          'dataset' + '_' + args.dataset + '_' + \
          'seed' + str(args.manual_seed)

args.checkpoint_dir = os.path.join(args.checkpoint_dir, log_dir)
if not os.path.isdir(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

if args.resume is False and args.evaluate is False:
    with open(log_txt, 'a+') as f:
        f.write("==========\nArgs:{}\n==========".format(args) + '\n')

np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
torch.set_printoptions(precision=4)

num_classes = 100
trainset = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                  [0.2675, 0.2565, 0.2761])
                                         ]))

testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                 [0.2675, 0.2565, 0.2761]),
                                        ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                          pin_memory=(torch.cuda.is_available()))

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                         pin_memory=(torch.cuda.is_available()))

print('==> Building model..')
net = getattr(models, args.arch)(num_classes=num_classes)
net.eval()
resolution = (1, 3, 32, 32)
print('Arch: %s, Params: %.2fM, Multi-adds: %.2fG'
      % (args.arch, cal_param_size(net) / 1e6, cal_multi_adds(net, resolution) / 1e9))
del (net)

# 创建模型 - 现在使用WideResNetDistill
model = getattr(models, args.arch)
net = model(num_classes=num_classes).cuda()

# 优化器 - 只优化主模型，蒸馏模型没有额外参数
optimizer = optim.SGD(net.parameters(), lr=args.init_lr,
                      momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

if args.freezed:
    print('load pre-trained backbone weights from: {}'.format(args.pretrained_backbone))
    checkpoint = torch.load(args.pretrained_backbone, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['state_dict'])
    for p in net.parameters():
        p.requires_grad = False
    net.eval()

net = torch.nn.DataParallel(net)
ss_feats, logits = net(torch.randn(2, 3, 32, 32))
num_ss_feats = len(ss_feats)
cudnn.benchmark = True


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        return loss


def correct_num(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum()
        res.append(correct_k)
    return res


def adjust_lr(optimizer, epoch, args, step=0, all_iters_per_epoch=0, eta_min=0.):
    cur_lr = 0.
    if epoch < args.warmup_epoch:
        cur_lr = args.init_lr * float(1 + step + epoch * all_iters_per_epoch) / (
                args.warmup_epoch * all_iters_per_epoch)
    else:
        epoch = epoch - args.warmup_epoch
        cur_lr = args.init_lr * 0.1 ** bisect_right(args.milestones, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr


def train(epoch, criterion_list, optimizer):
    train_loss = 0.
    train_loss_cls = 0.

    top1_num = 0
    top5_num = 0
    total = 0

    if epoch >= args.warmup_epoch:
        lr = adjust_lr(optimizer, epoch, args)

    start_time = time.time()
    criterion_cls = criterion_list[0]

    if not args.freezed:
        net.train()
    else:
        net.eval()

    for batch_idx, (input, target) in enumerate(trainloader):
        batch_start_time = time.time()
        input = input.float().cuda()
        target = target.cuda()

        if epoch < args.warmup_epoch:
            lr = adjust_lr(optimizer, epoch, args, batch_idx, len(trainloader))

        optimizer.zero_grad()

        # 使用WideResNetDistill返回的特征和logits
        ss_feats, logits = net(input)

        # 只使用主分类器的损失
        loss_cls = criterion_cls(logits, target)

        if not args.freezed:
            loss_cls.backward()
            optimizer.step()

        train_loss += loss_cls.item() / len(trainloader)
        train_loss_cls += loss_cls.item() / len(trainloader)

        # 计算准确率
        top1, top5 = correct_num(logits, target, topk=(1, 5))
        top1_num += top1
        top5_num += top5
        total += target.size(0)

        print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
            epoch, batch_idx, len(trainloader), lr, time.time() - batch_start_time, (top1_num / total).item()))

    acc1 = round((top1_num / total).item(), 4)
    acc5 = round((top5_num / total).item(), 4)

    print(f'Train epoch:{epoch}\nTrain Top-1 Acc: {acc1}\nTrain Top-5 Acc: {acc5}\n')

    with open(log_txt, 'a+') as f:
        f.write('Epoch:{}\t lr:{:.4f}\t duration:{:.3f}'
                '\n train_loss:{:.5f}\t train_loss_cls:{:.5f}'
                '\nTrain Top-1 Acc: {}\nTrain Top-5 Acc: {}\n'
                .format(epoch, lr, time.time() - start_time,
                        train_loss, train_loss_cls,
                        acc1, acc5))


def test(epoch, criterion_cls):
    global best_acc
    test_loss_cls = 0.

    top1_num = 0
    top5_num = 0
    total = 0

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(testloader):
            batch_start_time = time.time()
            input, target = inputs.cuda(), target.cuda()

            # 使用WideResNetDistill返回的特征和logits
            ss_feats, logits = net(input)

            # 只使用主分类器的损失
            loss_cls = criterion_cls(logits, target)
            test_loss_cls += loss_cls.item() / len(testloader)

            # 计算准确率
            top1, top5 = correct_num(logits, target, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += target.size(0)

            print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
                epoch, batch_idx, len(testloader), time.time() - batch_start_time, (top1_num / total).item()))

        acc1 = round((top1_num / total).item(), 4)
        acc5 = round((top5_num / total).item(), 4)

        with open(log_txt, 'a+') as f:
            f.write('test epoch:{}\t test_loss_cls:{:.5f}\nTop-1 Acc: {}\nTop-5 Acc: {}\n'
                    .format(epoch, test_loss_cls, acc1, acc5))

        print(f'test epoch:{epoch}\tTest Top-1 Acc: {acc1}\nTest Top-5 Acc: {acc5}\n')

    return acc1


if __name__ == '__main__':
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)

    if args.evaluate:
        print('load pre-trained weights from: {}'.format(
            os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar')))
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        test(start_epoch, criterion_cls)
    else:
        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_cls)  # classification loss
        criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
        criterion_list.cuda()

        if args.resume:
            print('load pre-trained weights from: {}'.format(
                os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar')))
            checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                    map_location=torch.device('cpu'))
            net.module.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1

        for epoch in range(start_epoch, args.epochs):
            train(epoch, criterion_list, optimizer)
            acc = test(epoch, criterion_cls)

            state = {
                'net': net.module.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'))

            is_best = False
            if best_acc < acc:
                best_acc = acc
                is_best = True

            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))

        print('Evaluate the best model:')
        print('load pre-trained weights from: {}'.format(
            os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar')))
        args.evaluate = True
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        top1_acc = test(start_epoch, criterion_cls)

        with open(log_txt, 'a+') as f:
            f.write('best_accuracy: {} \n'.format(best_acc))
        print('best_accuracy: {} \n'.format(best_acc))
        os.system('cp ' + log_txt + ' ' + args.checkpoint_dir)
