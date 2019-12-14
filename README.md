# CS454_project

## What we implemented
DefenseGAN using **Genetic Algorithm(GA)**, instead of Gradient descent(GD)
For more information, read our report : .

## How to execute
### Fast way
The easiest way to execute GA + DefenseGAN is to run 'wgan_torch/defenseGAN.py'. All you need to change in file is one line whether to run wGAN with GD or GA. We provides some checkpoints we used to train necessary models. Run in the root of your project:
```
>> cd wgan_torch
>> virtualenv -p python3 venv  (python version with 3.6 or 3.7 are all possible)
>> source venv/bin/activate
>> (venv)pip install -r requirements.txt
>> (venv)python defenseGan.py
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


## Methods
We tried various methods for the expirement.

#### 1. GD(Gradient Descent)
To compare the effectiveness of our new methods, we need to implement the original method described in [DefenseGAN paper](https://arxiv.org/pdf/1805.06605.pdf). The original DefenseGAN use Gradient Descent with population 10 and iterate total 200 times.

#### 2. GA(Genetic Algorithm)
Alternate GD with GA.
- population : default value is 10. Also tried various number of population as parameter
- crossover : Roulette wheel sampling method
- mutation : Gaussian mutation
- generation : default value is 200. Also tried various number of generation as parameter

#### 3. Memetic GA
Iterate 100 times for (GA+GD). This will run like GA-GD-GA-GD-GA-GD...
Total iteration is 200. (100 times with GA and 100 times with GD)

#### 4. GA and GD
Use GA with 100 generation - other hyperparamter or methods are same as above GA, and then iterate GD 100 times. Total iteration is 200.

## Result
We experience with 2 threat levels: white- and black-box attacks.

White-box models assume that the attacker has complete knowledge of all the classifier parameters. Use same classifier to both run the result, and to create FGSM images.

Unlike white-box, black-box adversaries have no access to the classifier or defense parameter. Use differenc classifier to run the result, and to create FGSM images.

### Result 1 : white-box attack
| Classifier | Data | Method | Population | Iteration | Epsilon = 0.1 | Epsilon = 0.2 | Epsilon = 0.3 | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | A | GD | 10 | 200 | 88/100 | 93/100 | 89/100 | 90% |
| B | B | GD | 10 | 200 | 78/100 | 90/100 | 87/100 | 85% |
| C | C | GD | 10 | 200 | 79/100 | 87/100 | 86/100 | 84% |
| A | A | GA | 10 | 200 | -/100 | -/100 | -/100 | -% |
| B | B | GA | 10 | 200 | 40/100 | 60/100 | 61x/100 | 53.66% |
| C | C | GA | 50 | 100 | 52/100 | 58/100 | 62/100 | 57.33% |
| A | A | Memetic GA | 10 | 200 | -/100 | -/100 | -/100 | -% |
| B | B | Memetic GA | 10 | 200 | -/100 | -/100 | -/100 | -% |
| C | C | Memetic GA | 10 | 200 | -/100 | -/100 | -/100 | -% |
| A | A | GA and GD | 10 | 200 | 70/100 | 80/100 | 82/100 | 77.33% |
| B | B | GA and GD | 10 | 200 | 71/100 | 78/100 | 83/100 | 77.33% |
| C | C | GA and GD | 10 | 200 | -/100 | -/100 | -/100 | -% |

## Result 2 : black-box attack

| Classifier | Data | Method | Population | Iteration | Epsilon = 0.1 | Epsilon = 0.2 | Epsilon = 0.3 | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C | A | - | 10 | 200 | 32/100 | 37/100 | 22/100 | 30.33% |
| C | A | GD | 10 | 200 | 72/100 | 87/100 | 86/100 | 81.67% |
| C | A | GA | 10 | 200 | -/100 | -/100 | -/100 | -% |
| C | A | Memetic GA | 10 | 200 | 61/100 | 70/100 | 73/100 | 68% |
| C | A | GA and GD | 10 | 200 | -/100 | -/100 | -/100 | -% |

## Result 3 : other trials..
| Classifier | Data | Method | Population | Iteration | Epsilon = 0.1 | Epsilon = 0.2 | Epsilon = 0.3 | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | A | Memetic GA | 10 | 400 | 74/100 | 76/100 | 81/100 | 77% |
| B | B | Memetic GA | 10 | 400 | 63/100 | 73/100 | 82/100 | 72.67% |
| C | C | Memetic GA | 10 | 400 | 75/100 | 76/100 | 77/100 | 76% |
| C | A | Memetic GA | 30 | 200 | 65/100 | 76/100 | 79/100 | 73.33% |
| C | A | Memetic GA | 10 | 600 | 65/100 | 74/100 | 76/100 | 71.67% |

## DEAP for GA
DEAP version == 1.3.0


## Related works
### DefenseGAN
- You can read DefenseGAN paper in here [link](https://arxiv.org/pdf/1805.06605.pdf)
- You can also see codes in github repository [link](https://github.com/kabkabm/defensegan)


### FGSM (Fast Gradient Signed Method)
- You can easily follow FGSM with pytorch from here [link](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)

## Contributors

- [Daeseong Kim](https://github.com/scvgoe)
- [Kanghoon Lee](https://github.com/leehoon7)
- [Younseo Choi](https://github.com/Choiyounseo)
- [Junmo Cho](https://github.com/junmokane)
- [Donghyun Kim](https://github.com/donghyun932)
