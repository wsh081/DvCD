
# using the teacher network of the version of a frozen backbone 

python train_student_cifar.py \
    --tarch wrn_40_2_distill \
    --arch wrn_16_2_distill \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_wrn_40_2_distill_dataset_cifar100_seed0/wrn_40_2_distill_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 1

python train_student_cifar.py \
    --tarch vgg13_distill \
    --arch vgg8_distill \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_vgg13_distill_dataset_cifar100_seed0/vgg13_distill_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 0


python train_student_cifar.py --tarch wrn_40_2_distill --arch wrn_40_1_distill \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_wrn_40_2_distill_dataset_cifar100_seed0/wrn_40_2_distill_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 0



python train_student_cifar.py --tarch resnet32x4_distill --arch resnet8x4_distill \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_resnet32x4_distill_dataset_cifar100_seed0/resnet32x4_distill_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 0

python train_student_cifar.py --tarch resnet56_distill --arch resnet20_distill \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_resnet56_distill_dataset_cifar100_seed0/resnet56_distill_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 0

python train_student_cifar.py --tarch wrn_40_2_distill --arch ShuffleV1_distill \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_wrn_40_2_distill_dataset_cifar100_seed0/wrn_40_2_distill_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --init-lr 0.01 \
    --gpu 0 --manual 0

python train_student_cifar.py --tarch resnet32x4_distill --arch resnet8x4_distill \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_resnet32x4_distill_dataset_cifar100_seed0/resnet32x4_distill_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 0

python train_student_cifar.py --tarch resnet56_distill --arch resnet20_distill \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_resnet56_distill_dataset_cifar100_seed0/resnet5_distill_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 0

python train_student_cifar.py --tarch vgg13_distill --arch mobilenetV2_distill \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_vgg13_distill_dataset_cifar100_seed0/vgg13_distill_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --init-lr 0.01 \
    --gpu 0 --manual 0



python train_student_cifar.py --tarch ResNet50_distill --arch mobilenetV2_distill \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_ResNet50_distill_dataset_cifar100_seed0/ResNet50_distill_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --init-lr 0.01 \
    --gpu 0 --manual 0



python train_student_cifar.py --tarch ResNet50_distill --arch vgg8_distill \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_ResNet50_distill_dataset_cifar100_seed0/ResNet50_distill_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --init-lr 0.01 \
    --gpu 0 --manual 0


python train_student_cifar.py --tarch resnet32x4_distill --arch ShuffleV2_distill \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_resnet32x4_distill_dataset_cifar100_seed0/resnet32x4_distill_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --init-lr 0.01 \
    --gpu 0 --manual 0
python train_student_cifar.py --tarch resnet32x4_distill --arch ShuffleV1_distill \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_resnet32x4_distill_dataset_cifar100_seed0/resnet32x4_distill_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --init-lr 0.01 \
    --gpu 0 --manual 0






python eval_rep.py \
    --arch wrn_16_2_distill \
    --dataset STL-10 \
    --data ./data/  \
    --s-path ./checkpoint/train_student_cifar_tarch_wrn_40_2_distill_arch_wrn_16_2_distill_dataset_cifar100_seed0/student_distill_best.pth.tar

python eval_rep.py \
    --arch wrn_16_2_distill \
    --dataset TinyImageNet \
    --data ./data/tiny-imagenet-200/ \
    --s-path ./checkpoint/train_student_cifar_tarch_wrn_40_2_distill_arch_wrn_16_2_distill_dataset_cifar100_seed0/student_distill_best.pth.tar


python eval_rep.py \
    --arch mobilenetV2_distill \
    --dataset tinyimagenet \
    --data ./data/tiny-imagenet-200/  \
    --s-path ./checkpoint/train_student_cifar_tarch_vgg13_distill_arch_mobilenetV2_distill_dataset_cifar100_seed0/student_distill_best.pth.tar


