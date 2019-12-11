# CS454_project

## What we implemented
DefenseGAN using **Genetic Algorithm(GA)**, instead of Gradient descent(GD)

## How to execute
### Fast way
The easiest way to execute GA + DefenseGAN is to run 'wgan_torch/defenseGAN.py'. All you need to change in file is one line whether to run wGAN with GD or GA. We provides some checkpoints we used to train necessary models. Run in the root of your project: 
```
>> cd wgan_torch
>> virtualenv -p python3 venv  (python version with 3.6 or 3.7 are all possible)
>> source venv/bin/activate
>> pip install -r requirements.txt
>> python defenseGan.py
```

### Follow from bottom to top
#### 1. Train basic classifier models
Train basic classifier models. All codes are ipynb file format, so that we recommend you to use google colab or jupyter for run. You can observe each model's form from `wgan_torch/classifiers/*.py`. You can train by yourself by using the codes of each model : `classifier_*/mnist_classifier_*.ipynb`.

To skip this step, just use provided checkpoints of each model. You can get each checkpoints from `wgan_torch/classifiers/*.pt`.

#### 2. Create adversarial images using FGSM
Run FGSM(Fast Gradient Signed Method) to create adversarial attack images. All codes are ipynb file format, so that we recommend you to use google colab or jupyter for run. Run `classifier_*/[fgsm_file].ipynb`.

To skip this step, just use provided FGSM images of each model. You can get each data from `wgan_torch/data/classifier_*` folders.

#### 3. Train wGAN models


#### 4. wGAN + (GA or GD)
Follow the above **Fast way** section explanation.

## Result
#### Gradient Descent
| Type | Epsilon = 0.1 | Epsilon = 0.2 | Epsilon = 0.3 | Total |
| --- | --- | --- | --- | --- |
| Classifier A | 88/100 | 93/100 | 89/100 | 90% |
| Classifier B | 78/100 | 90/100 | 87/100 | 85% |
| Classifier C | 79/100 | 87/100 | 86/100 | 84% |

#### Memetic Genetic Algorithm
| Type | Epsilon = 0.1 | Epsilon = 0.2 | Epsilon = 0.3 | Total |
| --- | --- | --- | --- | --- |
| Classifier A | 74/100 | 76/100 | 81/100 | 77% |
| Classifier B | - | - | - | - |
| Classifier C | - | - | - | - |

## DEAP for GA
DEAP version == 1.3.0

## Related works
### DefenseGAN
You can read DefenseGAN paper in here [link](https://arxiv.org/pdf/1805.06605.pdf)

You can also see codes in github repository [link](https://github.com/kabkabm/defensegan)

### FGSM (Fast Gradient Signed Method)
You can easily follow FGSM with pytorch from here [link](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)

## Contributors

- [Daeseong Kim](https://github.com/scvgoe)
- [Kanghoon Lee](https://github.com/leehoon7)
- [Younseo Choi](https://github.com/Choiyounseo)
- [Junmo Cho](https://github.com/junmokane)
- [Donghyun Kim](https://github.com/donghyun932)