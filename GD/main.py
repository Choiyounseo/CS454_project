import torch
import torch.nn as nn
import torchvision.utils as utils
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from model import Generator

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
    # todo R times doing this -> optimal z will be z hat
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    L = 12000
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
fgsm_image_path = './data/fgsm_images/0.25_8_to_3_885.jpg'
model_weight_path = './data/weights_/netG_12500.pth'

# Generator(ngpu, nc, nz, ngf)
netG = Generator(params['ngpu'], params['nc'], params['nz'], params['ngf'])
netG.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

fgsm_image=mpimg.imread(fgsm_image_path)
# plt.imshow(fgsm_image, cmap='gray')  # plot the image of fgsm_image

#########################
# Do the GD from here!! # or gradient descent
#########################
# Thought : manipulating latent vectors is important since domain is specified (z is normal dist.)
# and, GA should have high converging power.

fgsm_image = fgsm_image.reshape(1,fgsm_image.shape[0],fgsm_image.shape[1])
fgsm_image = torch.from_numpy(fgsm_image).float()
print(fgsm_image.shape)  # torch.Size([1, 28, 28])
result = defensegan(fgsm_image)
# imshow(fgsm_image.detach())  # plot the image of fgsm image
imshow(result.detach())  # plot the image of generated image
