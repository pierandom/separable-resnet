# SeparableResNet

Experimenting with *Deep Learning* ideas and *Convolutional Neural Networks* on my laptop.

## Network Architecture
The network architecture employs **skip connections** as in [*ResNet*](https://arxiv.org/abs/1512.03385) and **depthwise separable** convolutions as in [*MobileNet*](https://arxiv.org/abs/1704.04861).

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

## Results on CIFAR10
Model format is: *SeparableResNet\<width-factor>-\<depth-factor>*

|Model             |Parameters|Test Accuracy|
|     :---:        |   :---:  |    :---:    |
|SeparableResNet4-3|   0.45M  |    96.49    |
| ResNet32*        |   0.47M  |    94.19    | 

\*This implementation employs shortcut connections of type (B) instead of type (A). It also follows the above training recipe.

Weights are available under `trained-models`. Look at `check_results.py` on how to use them.

### Test Accuracy Breakdown
| Epoch | SeparableResNet4-3 | ResNet32 |
| :---: | :---: | :---: |
| 30    | 92.28 | 89.74 |
| 60    | 93.34 | 91.19 |
| 120   | 94.56 | 92.61 |
| 240   | 95.00 | 93.55 |
| 480   | 95.65 | 94.04 |
| 960   | 96.09 | 94.19 |
| 1920  | 96.49 |   -   |

- From the progression it looks like further improvements are possible for *SeparableResNet4-3* but this becomes exponentially difficult.

- We can see this training recipe is more effective on *ResNet32* than the original one.

- Improvements are steadier for *SeparableResNet4-3*.

![Test Accuracy learning curve](trained-models/CIFAR10/separable-resnet4-3/test-accuracy.png)