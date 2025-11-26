
# Dual-path collaborative distillation
 - This project provides source code for our Dual-view collaborative distillation (DvCD).

## Installation

### Requirements


Python 3.8 ([Anaconda](https://www.anaconda.com/) is recommended)

CUDA 12.1

PyTorch 2.1.1


## Perform  experiments on CIFAR-100 dataset
#### Dataset
CIFAR-100 : [download](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

unzip to the `./data` folder

#### Training baselines
```
python train_baseline_cifar.py --arch wrn_16_2 --data ./data/  --gpu 0
```
More commands for training various architectures can be found in [train_baseline_cifar.sh]

#### Training teacher networks

You can use the following commands to train your own teacher network.
```
python train_teacher_cifar.py \
    --arch wrn_40_2_distill \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 0
```

#### Training student networks
(1) train baselines of student networks
```
python train_baseline_cifar.py --arch wrn_16_2 --data ./data/  --gpu 0
```


(2) train student networks with a pre-trained teacher network

Note that the specific teacher network should be pre-trained before training the student networks

python train_student_cifar.py \
    --tarch wrn_40_2_distill \
    --arch wrn_16_2_distill \
    --tcheckpoint ./checkpoint/train_teacher_cifar_arch_wrn_40_2_distill_dataset_cifar100_seed0/wrn_40_2_distill_best.pth.tar \
    --checkpoint-dir ./checkpoint \
    --data ./data \
    --gpu 0 --manual 1
```

More commands for training various teacher-student pairs can be found in [train_student_cifar.sh]

####  Results of the same architecture style between teacher and student networks

|Teacher <br> Student | WRN-40-2 <br> WRN-16-2 | ResNet32×4  <br> ResNet8×4 | ResNet-56 <br> ResNet-20 | WRN-40-2  <br> WRN-40-1 | VGG13<br> VGG8 |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:--------------------:|:--------------------:|
| Teacher  |    75.61 | 79.42 | 72.34 | 75.61 | 74.64 |
| Student | 73.26| 72.50| 69.06| 71.98| 70.36 |
| DPCD | 77.02| 77.28| 72.69| 75.63| 75.79|
 


####  Results of different architecture styles between teacher and student networks

|Teacher <br> Student |ResNet32×4  <br>ShufffeNetV2  |   VGG13  <br> MobileNetV2 | ResNet-50 <br> MobileNetV2 | WRN-40-2<br> ShuffleNetV1 |
|:---------------:|:-----------------:|:-----------------:|:--------------------:|:--------------------:|
| Teacher  |    79.42|74.64 |79.34 |75.61   |
| Student | 71.82|  64.60 |64.60| 70.50 |
| DPCD | 78.09  | 70.81 | 70.90  |77.72 |






 
 
