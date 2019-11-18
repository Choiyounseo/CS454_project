import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from model import Generator, Discriminator
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def imshow(img):
    img = (img + 1) / 2
    img = img.squeeze()
    np_img = img.numpy()
    plt.imshow(np_img, cmap='gray')
    plt.show()

def imshow_grid(img):
    img = utils.make_grid(img.cpu().detach())
    img = (img + 1) / 2
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Hyper parameters
params = {
    'input_size': 28,  # image size 1x64x64
    'batch_size': 64,  # batch size
    'pop_size': 100,   # population size
    'nc': 1,  # number of channels
    'nz': 100,  # size of z latent vector
    'ngf': 64,  # size of feature maps in generator
    'ndf': 32,  # size of feature maps in discriminator
    'num_epochs': 1000,  # number of epochs
    'lr': 0.0001,  # learning rate
    'beta1': 0.5,   # beta1 for adam optimizer
    'ngpu': 1,  # number of GPU
    'lambda_gp': 10,  # loss weight for gradient penalty
    'n_critic': 5,
}

# Generator(ngpu, nc, nz, ngf)
netG = Generator(params['ngpu'], params['nc'], params['nz'], params['ngf'])
netG.load_state_dict(torch.load('./data/weights_/netG_12500.pth'))

# transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,), std=(0.5,))])
# data sets and data loader
train_data = datasets.MNIST(root='data/', train=True, transform=transform, download=True)
train_data_loader = DataLoader(train_data, params['batch_size'], shuffle=False)
first_batch = train_data_loader.__iter__().__next__()  # first batch of MNIST data set : torch.Size([64x, 1, 28, 28])
print(first_batch[0][0].shape)  # torch.Size([1, 28, 28])
#imshow(first_batch[0][0])  # plot the image of first batch

# Input image for defense GAN
# For the test purpose, we will use MNIST data sample first.
# fgsm_image : torch.Size([1, 28, 28]). This is image x.
fgsm_image = first_batch[0][0]  # torch.Size([1, 28, 28]). This should be fgsm_image later on.

# Initial population for GA
# initial_population : torch.Size([100, 100, 1, 1]), This has 100 latent vectors z (z is torch.Size([100, 1, 1])).
# for example, initial_population[0] is z_0, initial_population[1] is z_1, ..., initial_population[99] is z_99.
initial_population = torch.FloatTensor(params['pop_size'], params['nz'], 1, 1).normal_(0, 1)
print(initial_population.shape)  # torch.Size([100, 100, 1, 1])

# generated images from latent vectors z.
# gen_images : torch.Size([100, 1, 28, 28]). This is generated images from initial_population.
# This has 100 images G(z). (G(z) is torch.Size([1, 28, 28]).
# for example, gen_images[0] is G(z_0), gen_images[1] is G(z_1), ..., gen_images[99] is G(z_99).
gen_images = netG(initial_population)
#imshow_grid(gen_images.view(-1, 1, 28, 28))  # plot the image of generated images

#########################
# Do the GA from here!! # or gradient descent
#########################
# Thought : manipulating latent vectors is important since domain is specified (z is normal dist.)
# and, GA should have high converging power.

# For each generation, select the latent vector z* that minimizes fitness, and do the following.
z = torch.FloatTensor(1, params['nz'], 1, 1).normal_(0, 1)  # torch.Size([100, 1, 1]). This should be z* later on.
print("the shape of latent vector : " + str(z.shape))
gen_image = netG(z)  # torch.Size([1, 28, 28]). This is the generated image that we want to see for each generation.
# Because gen_image should step closer to fgsm_image x for each generation.
print("the shape of generated image : " + str(gen_image.shape))
imshow(gen_image.detach())  # plot the image of generated image

# After GA, give generated image as input to each classifier (use gen_image)
