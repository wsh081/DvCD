
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import shutil
import argparse
import numpy as np
import time
import math
import models
import torchvision
import torchvision.transforms as transforms
from bisect import bisect_right
import matplotlib.pyplot as plt
import numpy as np

# 数据变换
def get_transforms():
    """创建弱增强和强增强变换"""
    # 弱增强：随机裁剪 + 水平翻转
    weak_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    ])


    # 强增强：在弱增强基础上添加更多扰动
    strong_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    ])

    # 测试变换
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    ])

    return weak_transform, strong_transform, test_transform


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--data', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset name')
parser.add_argument('--arch', default='wrn_16_distill', type=str, help='student network architecture')
parser.add_argument('--tarch', default='wrn_40_distill', type=str, help='teacher network architecture')
parser.add_argument('--tcheckpoint', default='wrn_40_2_distill.pth.tar', type=str,
                    help='pre-trained weights of teacher')
parser.add_argument('--init-lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[150, 180, 210], type=list, help='milestones for lr-multistep')
parser.add_argument('--sgdr-t', default=300, type=int, dest='sgdr_t', help='SGDR T_0')
parser.add_argument('--warmup-epoch', default=0, type=int, help='warmup epoch')
parser.add_argument('--epochs', type=int, default=240, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='the number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--kd_T', type=float, default=3, help='temperature for KD distillation')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='directory fot storing checkpoints')
parser.add_argument('--infomax-alpha', type=float, default=1.0, help='weight for InfoMax loss')
parser.add_argument('--infomax-T', type=float, default=3.0, help='temperature for InfoMax loss')
parser.add_argument('--consistency-beta', type=float, default=1.0, help='weight for consistency loss')
parser.add_argument('--consistency-T', type=float, default=3.0, help='temperature for consistency loss')

# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

log_txt = 'result/' + str(os.path.basename(__file__).split('.')[0]) + '_' + \
          'tarch' + '_' + args.tarch + '_' + \
          'arch' + '_' + args.arch + '_' + \
          'dataset' + '_' + args.dataset + '_' + \
          'seed' + str(args.manual_seed) + '.txt'

log_dir = str(os.path.basename(__file__).split('.')[0]) + '_' + \
          'tarch' + '_' + args.tarch + '_' + \
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

# 获取增强变换
weak_transform, strong_transform, test_transform = get_transforms()


# 创建增强数据集
class DualTransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, weak_transform, strong_transform):
        self.dataset = dataset
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        img_weak = self.weak_transform(image)
        img_strong = self.strong_transform(image)
        return img_weak, img_strong, label

    def __len__(self):
        return len(self.dataset)


# 原始数据集
original_trainset = torchvision.datasets.CIFAR100(
    root=args.data, train=True, download=True
)

# 创建双变换数据集
trainset = DualTransformDataset(original_trainset, weak_transform, strong_transform)

testset = torchvision.datasets.CIFAR100(
    root=args.data, train=False, download=True,
    transform=test_transform
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True,
    pin_memory=(torch.cuda.is_available()), num_workers=args.num_workers
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False,
    pin_memory=(torch.cuda.is_available())
)

print('==> Building model..')
# 使用蒸馏模型
tnet = getattr(models, args.tarch)(num_classes=num_classes)
tnet.eval()

print('Teacher Arch: %s' % args.tarch)

# 使用蒸馏模型
net = getattr(models, args.arch)(num_classes=num_classes)
net.eval()
print('Student Arch: %s' % args.arch)

print('load pre-trained teacher weights from: {}'.format(args.tcheckpoint))
checkpoint = torch.load(args.tcheckpoint, map_location=torch.device('cpu'))

# 加载教师模型
tnet = tnet.cuda()
tnet.load_state_dict(checkpoint['net'])
tnet.eval()
tnet = torch.nn.DataParallel(tnet)

# 加载学生模型
net = net.cuda()
net = torch.nn.DataParallel(net)

cudnn.benchmark = True


# ================== 统一的一致性损失函数 ==================
class UnifiedConsistencyLoss(nn.Module):
    """
    统一的融合损失函数
    结合了InfoMax损失和一致性损失的优势

    参数:
        T (float): 温度参数
        alpha (float): InfoMax损失权重
        beta (float): 一致性损失权重
    """

    def __init__(self, T=3.0, alpha=1.0, beta=1.0):
        super(UnifiedConsistencyLoss, self).__init__()
        self.alpha = alpha  # InfoMax损失权重
        self.beta = beta  # 一致性损失权重
        self.T = T
        self.teacher_ratio = 0.9  # 默认教师比例

    def set_teacher_ratio(self, ratio):
        """动态设置教师分布的融合比例 (0.0-1.0)"""
        self.teacher_ratio = ratio

    def forward(self, student_logits_weak, student_logits_strong, teacher_logits_weak, teacher_logits_strong):
        """
        计算统一的融合损失

        参数:
            student_logits_weak (Tensor): 学生对弱增强视图的logits输出
            student_logits_strong (Tensor): 学生对强增强视图的logits输出
            teacher_logits_weak (Tensor): 教师对弱增强视图的logits输出
            teacher_logits_strong (Tensor): 教师对强增强视图的logits输出

        返回:
            Tensor: 统一的融合损失值
        """
        # 1. 计算InfoMax损失部分 -------------------------------
        # 计算弱增强视图的融合分布
        teacher_probs_weak = F.softmax(teacher_logits_weak / self.T, dim=1)
        student_probs_weak = F.softmax(student_logits_weak / self.T, dim=1)
        joint_probs_weak = teacher_probs_weak * self.teacher_ratio + student_probs_weak * (1 - self.teacher_ratio)



        # 计算强增强视图的融合分布
        teacher_probs_strong = F.softmax(teacher_logits_strong / self.T, dim=1)
        student_probs_strong = F.softmax(student_logits_strong / self.T, dim=1)
        joint_probs_strong = teacher_probs_strong * self.teacher_ratio + student_probs_strong * (1 - self.teacher_ratio)



        # 学生输出的对数概率
        student_log_probs_weak = F.log_softmax(student_logits_weak / self.T, dim=1)
        student_log_probs_strong = F.log_softmax(student_logits_strong / self.T, dim=1)

        # 计算InfoMax损失
        loss_infomax_weak = F.kl_div(student_log_probs_weak, joint_probs_weak.detach(), reduction='batchmean') * (
                    self.T ** 2)
        loss_infomax_strong = F.kl_div(student_log_probs_strong, joint_probs_strong.detach(), reduction='batchmean') * (
                    self.T ** 2)
        loss_infomax = (loss_infomax_weak + loss_infomax_strong)

        # 2. 计算一致性损失部分 -------------------------------
        # 学生自一致性损失
        p_weak = F.softmax(student_logits_weak / self.T, dim=1).detach()
        p_strong = F.log_softmax(student_logits_strong / self.T, dim=1)
        loss_self_consistency = F.kl_div(p_strong, p_weak, reduction='batchmean') * (self.T ** 2)

        # 教师-学生一致性损失
        p_teacher = F.softmax(teacher_logits_weak / self.T, dim=1).detach()
        loss_teacher_consistency = F.kl_div(p_strong, p_teacher, reduction='batchmean') * (self.T ** 2)

        # 总一致性损失
        loss_consistency = (loss_self_consistency + loss_teacher_consistency)

        # 3. 组合损失
        total_loss = self.alpha * loss_infomax + self.beta * loss_consistency
        # total_loss = self.alpha * loss_infomax
        # total_loss =  self.beta * loss_consistency
        return total_loss


# ================== 蒸馏损失函数 ==================
class DistillationLoss(nn.Module):
    """知识蒸馏损失（KL散度）"""

    def __init__(self, T=3.0):
        super(DistillationLoss, self).__init__()
        self.T = T  # 温度参数

    def forward(self, student_logits, teacher_logits):
        """
        计算蒸馏损失

        参数:
            student_logits (Tensor): 学生模型的logits输出
            teacher_logits (Tensor): 教师模型的logits输出

        返回:
            Tensor: KL散度损失值
        """
        soft_teacher = F.softmax(teacher_logits / self.T, dim=1).detach()
        log_soft_student = F.log_softmax(student_logits / self.T, dim=1)
        return F.kl_div(log_soft_student, soft_teacher, reduction='batchmean') * (self.T ** 2)

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

# 定义DKDloss类
class DKDloss(nn.Module):
    def __init__(self,warmup=20):
        super(DKDloss, self).__init__()
        self.warmup = warmup

    # 前向传播
    def forward(self, logits_student, logits_teacher, target,epoch, alpha=1.0, beta=4.0, temperature=3.0):
        # 获取ground truth mask和非ground truth mask
        gt_mask = _get_gt_mask(logits_student, target)
        other_mask = _get_other_mask(logits_student, target)

        # 对学生和教师logits进行softmax
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

        # 拼接mask
        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)

        # 计算tckd_loss
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
                F.kl_div(log_pred_student, pred_teacher, reduction='sum')
                * (temperature ** 2) / target.shape[0]
        )

        # 计算nckd_loss
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
                F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
                * (temperature ** 2) / target.shape[0]
        )

        # 计算总的loss
        loss = alpha * tckd_loss + beta * nckd_loss
        loss = min(epoch / self.warmup, 1.0) * loss
        return loss

def correct_num(output, target, topk=(1,)):
    """计算指定k值的准确率@k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k)
    return res

#
def adjust_lr(optimizer, epoch, args):
    """调整学习率"""
    if epoch < args.warmup_epoch:
        # 线性预热
        lr = args.init_lr * (epoch + 1) / args.warmup_epoch
    else:
        # 多步下降
        epoch = epoch - args.warmup_epoch
        lr = args.init_lr * 0.1 ** bisect_right(args.milestones, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(epoch, criterion_list, optimizer):
    train_loss = 0.
    train_loss_cls = 0.
    train_loss_kd = 0.
    train_loss_unified = 0.
    top1_num = 0
    top5_num = 0
    total = 0

    # 在每个epoch开始时调整学习率
    lr = adjust_lr(optimizer, epoch, args)

    start_time = time.time()
    criterion_cls = criterion_list[0]
    criterion_kd = criterion_list[1]  # KL损失
    criterion_unified = criterion_list[2]  # 统一一致性损失

    # 动态调整教师分布的融合比例
    if epoch < 150:
        teacher_ratio = 0.9
    elif epoch < 180:
        teacher_ratio = 0.7
    else:
        teacher_ratio = 1.0
# wrn162 vgg8
#     if epoch < 150:
#         teacher_ratio = 1.0
#     elif epoch < 180:
#         teacher_ratio = 0.9
#     elif epoch < 210:
#         teacher_ratio = 0.8
#     else:
#         teacher_ratio = 1.0

    criterion_unified.set_teacher_ratio(teacher_ratio)
#wrn401
    net.train()

    for batch_idx, (inputs_weak, inputs_strong, target) in enumerate(trainloader):
        batch_start_time = time.time()
        inputs_weak = inputs_weak.float().cuda()
        inputs_strong = inputs_strong.float().cuda()
        target = target.cuda()

        optimizer.zero_grad()

        # 教师模型前向传播
        with torch.no_grad():
            _, teacher_logits_weak = tnet(inputs_weak)
            _, teacher_logits_strong = tnet(inputs_strong)

        # 学生模型前向传播
        _, student_logits_weak = net(inputs_weak)
        _, student_logits_strong = net(inputs_strong)

        # 分类损失（弱增强输出）
        loss_cls = criterion_cls(student_logits_weak, target)

        # 蒸馏损失（教师弱增强输出 vs 学生弱增强输出）
        loss_kd = criterion_kd(student_logits_weak, teacher_logits_weak.detach())

        # 统一一致性损失
        loss_unified = criterion_unified(
            student_logits_weak, student_logits_strong,
            teacher_logits_weak.detach(), teacher_logits_strong.detach()
        )

        # 总损失：分类损失 + KD损失 + 统一一致性损失
        loss = loss_cls + loss_kd + loss_unified
        # loss = loss_cls + loss_kd
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)

        loss.backward()
        optimizer.step()

        # 损失统计
        train_loss += loss.item()
        train_loss_cls += loss_cls.item()
        train_loss_kd += loss_kd.item()
        train_loss_unified += loss_unified.item()

        # 准确率统计
        top1, top5 = correct_num(student_logits_weak, target, topk=(1, 5))
        top1_num += top1.item()
        top5_num += top5.item()
        total += target.size(0)

        if batch_idx % 50 == 0:
            current_acc = top1_num / total * 100.0
            # 打印信息
            print(f"Epoch: {epoch}, Batch: {batch_idx}/{len(trainloader)}, "
                  f"LR: {lr:.5f}, Duration: {time.time() - batch_start_time:.2f}, "
                  f"Top-1 Acc: {current_acc:.4f}%, "
                  f"Teacher Ratio: {teacher_ratio:.4f}, "
                  f"Unified Loss: {loss_unified.item():.4f}")

    # 计算平均损失和准确率
    avg_loss = train_loss / len(trainloader)
    avg_loss_cls = train_loss_cls / len(trainloader)
    avg_loss_kd = train_loss_kd / len(trainloader)
    avg_loss_unified = train_loss_unified / len(trainloader)
    acc1 = top1_num / total * 100.0
    acc5 = top5_num / total * 100.0

    # 打印训练结果
    print(f"Train Epoch: {epoch}, Loss: {avg_loss:.4f}, Top-1: {acc1:.4f}%, Top-5: {acc5:.4f}%, "
          f"Teacher Ratio: {teacher_ratio:.4f}, Unified Loss: {avg_loss_unified:.4f}")

    # 记录日志
    with open(log_txt, 'a+') as f:
        f.write(
            f"Epoch: {epoch}\t LR: {lr:.5f}\t Duration: {time.time() - start_time:.3f}\t Teacher Ratio: {teacher_ratio:.4f}\n"
            f"Train Loss: {avg_loss:.5f}\t Cls Loss: {avg_loss_cls:.5f}\t KD Loss: {avg_loss_kd:.5f}\t "
            f"Unified Loss: {avg_loss_unified:.5f}\n"
            f"Train Top-1 Acc: {acc1:.4f}%\t Train Top-5 Acc: {acc5:.4f}%\n")

    return acc1


def test(epoch, net):
    global best_acc
    test_loss_cls = 0.
    top1_num = 0
    top5_num = 0
    total_samples = 0

    criterion_cls = nn.CrossEntropyLoss()

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(testloader):
            batch_start_time = time.time()
            inputs, target = inputs.cuda(), target.cuda()

            # 学生模型前向传播
            _, student_logits = net(inputs)

            # 分类损失
            loss_cls = criterion_cls(student_logits, target)
            test_loss_cls += loss_cls.item()

            # 计算准确率
            top1, top5 = correct_num(student_logits, target, topk=(1, 5))
            top1_num += top1.item()
            top5_num += top5.item()
            total_samples += target.size(0)

            if batch_idx % 50 == 0:
                current_acc = top1_num / total_samples * 100.0
                print(f"Test | Epoch: {epoch} | Batch: {batch_idx}/{len(testloader)} | "
                      f"Top-1 Acc: {current_acc:.4f}%")

    # 计算总准确率
    acc1 = top1_num / total_samples * 100.0
    acc5 = top5_num / total_samples * 100.0
    avg_loss = test_loss_cls / len(testloader)

    print(f"Test | Epoch: {epoch} | Loss: {avg_loss:.4f} | Top-1: {acc1:.4f}% | Top-5: {acc5:.4f}%")

    # 记录日志
    with open(log_txt, 'a+') as f:
        f.write(f"Test | Epoch: {epoch} | Loss: {avg_loss:.5f} | Top1: {acc1:.4f}% | Top5: {acc5:.4f}%\n")

    return acc1


if __name__ == '__main__':
    best_acc = 0.  # 最佳测试准确率
    start_epoch = 0  # 开始epoch（0或上次检查点结束的epoch）

    # 初始化损失函数
    criterion_cls = nn.CrossEntropyLoss()
    criterion_kd = DistillationLoss(T=args.kd_T)
    # criterion_kd = DKDloss()
    criterion_unified = UnifiedConsistencyLoss(
        T=args.infomax_T,
        alpha=args.infomax_alpha,
        beta=args.consistency_beta
    ).cuda()

    if args.evaluate:
        print('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir, 'student_distill.pth.tar')))
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'student_distill.pth.tar'),
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        test(start_epoch, net)
    else:
        # 测试教师模型
        print('Evaluating Teacher Model...')
        acc = test(0, tnet)
        print(f'Teacher Acc: {acc:.4f}%')

        # 初始化优化器
        optimizer = optim.SGD(
            net.parameters(),
            lr=args.init_lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True
        )

        criterion_list = nn.ModuleList([
            criterion_cls,  # 分类损失
            criterion_kd,  # 蒸馏损失
            criterion_unified  # 统一一致性损失
        ]).cuda()

        if args.resume:
            print('load pre-trained weights from: {}'.format(
                os.path.join(args.checkpoint_dir, 'student_distill.pth.tar')))
            checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'student_distill.pth.tar'),
                                    map_location=torch.device('cpu'))
            net.module.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1

        # 训练循环
        for epoch in range(start_epoch, args.epochs):
            # 训练一个epoch
            train_acc = train(epoch, criterion_list, optimizer)

            # 在测试集上评估
            test_acc = test(epoch, net)

            # 保存检查点
            state = {
                'net': net.module.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.checkpoint_dir, 'student_distill.pth.tar'))

            # 更新最佳模型
            if best_acc < test_acc:
                best_acc = test_acc
                shutil.copyfile(os.path.join(args.checkpoint_dir, 'student_distill.pth.tar'),
                                os.path.join(args.checkpoint_dir, 'student_distill_best.pth.tar'))
                print(f"New best model saved at epoch {epoch} with accuracy: {best_acc:.4f}%")

        # 评估最佳模型
        print('Evaluating Best Student Model...')
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'student_distill_best.pth.tar'),
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        test(checkpoint['epoch'], net)

        # 记录最终结果
        with open(log_txt, 'a+') as f:
            f.write(f'best_accuracy: {best_acc:.4f}% \n')
        print(f'Best accuracy achieved: {best_acc:.4f}%')
        os.system('cp ' + log_txt + ' ' + args.checkpoint_dir)
