
python train_teacher_cifar.py \
    --arch wrn_40_2_distill \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 0


python train_teacher_cifar.py \
    --arch ResNet50_distill \
    --checkpoint-dir ./checkpoint \
    --data ./data  \
    --gpu 0 --manual 0

python train_teacher_cifar.py \
    --arch vgg13_distill \
    --checkpoint-dir ./checkpoint \
    --data ./data  \
    --gpu 0 --manual 0

python train_teacher_cifar.py \
    --arch resnet56_distill \
    --checkpoint-dir ./checkpoint \
    --data ./data  \
    --gpu 0 --manual 0

python train_teacher_cifar.py \
    --arch resnet32x4_distill \
    --checkpoint-dir ./checkpoint \
    --data ./data  \
    --gpu 0 --manual 0
