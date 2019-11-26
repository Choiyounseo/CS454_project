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

def defensegan(x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    L = 1000
    lr = 500

    z = torch.randn((x.shape[0], 100)).view(-1, 100, 1, 1).to(device)
    z.requires_grad = True
    for l in range(L):
        print(l)
        samples = netG(z)
        MSE_loss = nn.MSELoss()
        loss_mse = MSE_loss(samples[0], x)
        loss_mse.backward()
        z = z - lr * z.grad
        z = z.detach()  # not leaf
        z.requires_grad = True

    return netG(z)

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


fixed_noise = torch.FloatTensor(params['batch_size'], params['nz'], 1, 1).normal_(0, 1)
# Generator(ngpu, nc, nz, ngf)
netG = Generator(params['ngpu'], params['nc'], params['nz'], params['ngf'])
netG.load_state_dict(torch.load('./data/weights/netG_10000.pth'))
gen_images = netG(fixed_noise)
print(gen_images.shape)
#imshow_grid(gen_images.view(-1, 1, 28, 28))
imshow(gen_images.detach()[0][0])

'''

# transform
transform = transforms.Compose([transforms.ToTensor()])
# data sets and data loader
train_data = datasets.MNIST(root='data/', train=True, transform=transform, download=True)
train_data_loader = DataLoader(train_data, params['batch_size'], shuffle=False)
first_batch = train_data_loader.__iter__().__next__()
print(first_batch[0][0].shape)
#imshow(first_batch[0][0])

# input image for defense GAN. size should be torch.Size([1, 28, 28])
fgsm_image = first_batch[0][0]

# initial population for GA
initial_population = torch.FloatTensor(params['pop_size'], params['nz'], 1, 1).normal_(0, 1)
print(initial_population.shape)
gen_images = netG(fixed_noise)
#imshow_grid(gen_images.view(-1, 1, 28, 28))

# do the GA
# thought : manipulating latent vectors is important since domain is specified
# , GA should be very converging
# for each generation, store the latent vector that minimizes fitness
z = torch.FloatTensor(1, params['nz'], 1, 1).normal_(0, 1)
print("the shape of latent vector : " + str(z.shape))
gen_image = netG(z)
print("the shape of generated image : " + str(gen_image.shape))
imshow(gen_image.detach())

# After GA, give generated image as input to each classifier (use gen_image)

'''

