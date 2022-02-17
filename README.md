# SeparableResNet

Experimenting with *Deep Learning* ideas and *Convolutional Neural Networks* on my laptop.

## Network Architecture
The network architecture employs **skip connections** as in *ResNet* and **depthwise separable** convolutions as in *MobileNet*.

## Training recipe
**Old tricks**
- Data Augmentation
    - Random Horizontal Flip
    - Trivial Augmentation Wide
    - Random Erasing
- Weight Decay

**New Tricks**
- Gradient Norm Clipping
- LR Schedule: Cosine Annealing with Restarts
- Label Smoothing

*Old tricks* is common practice whereas *new tricks* is not. 

## Results
Model format is: *SeparableResNet\<width-factor>-\<depth-factor>*

|Model             |Parameters|Test Accuracy|
|     :---:        |   :---:  |    :---:    |
|SeparableResNet4-3|   0.45M  |     96.09   |

Weights are available under `trained-models`.

### Training Breakdown
|  |  |  |  |  |  |  |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|**Epoch**   |   30  |   60  |  120  |  240  |  480  |  960  |
|**Accuracy**| 92.28 | 93.34 | 94.56 | 95.00 | 95.65 | 96.09 |

From the progression it looks like further improvements are possible but this would require many days of training on my laptop.

![Test Accuracy learning curve](trained-models/CIFAR10/separable-resnet4-3/test-accuracy.png)


### Insight
I think the good performances are due more to the training recipe than the network architecture.