import time
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torch
from torch import nn
import torch.optim as optim
from util import imshow, weights_init
from matplotlib import pyplot as plt
from model import Generator, Discriminator
from torch.autograd import Variable
import torch.autograd as autograd


# Hyper parameters
params = {
    'input_size': 28,  # image size 1x64x64
    'batch_size': 200,  # batch size
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

device = torch.device("cuda:0" if (torch.cuda.is_available() and params['ngpu'] > 0) else "cpu")
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,), std=(0.5,))])
# data sets and data loader
train_data = datasets.MNIST(root='data/', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='data/', train=False, transform=transform, download=True)
train_data_loader = DataLoader(train_data, params['batch_size'], shuffle=True)
test_data_loader = DataLoader(test_data, params['batch_size'], shuffle=False)

first_batch = train_data_loader.__iter__().__next__()
print('{:15s} | {:<25s} | {}'.format('name', 'type', 'size'))
print('{:15s} | {:<25s} | {}'.format('first_batch', str(type(first_batch)), len(first_batch)))
print('{:15s} | {:<25s} | {}'.format('first_batch[0]', str(type(first_batch[0])), first_batch[0].shape))
print('{:15s} | {:<25s} | {}'.format('first_batch[1]', str(type(first_batch[1])), first_batch[1].shape))


'''
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(first_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()
plt.close()

'''


# Generator(ngpu, nc, nz, ngf)
netG = Generator(params['ngpu'], params['nc'], params['nz'], params['ngf']).to(device)
''' this part can be used later when ngpu > 1
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
'''
netG.apply(weights_init)
print(netG)

# Discriminator(ngpu, nc, ndf)
netD = Discriminator(params['ngpu'], params['nc'], params['ndf']).to(device)
netD.apply(weights_init)
print(netD)

# Criterion & Optimizer
criterion = nn.BCELoss()
fixed_noise = torch.randn(64, params['nz'], 1, 1).to(device)
real_label = 1
fake_label = 0
optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Training Loop
img_list = []
G_losses = []
D_losses = []
D_x_set = []
D_G_z_set = []
mini_step = int(len(train_data) / (10 * params['batch_size']))
print(mini_step)
print("Starting Training Loop...")
since = time.time()
for epoch in range(params['num_epochs']):
    for i, data in enumerate(train_data_loader, 0):
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()
        # Train with real batches
        real = data[0].to(device)
        b_size = real.size(0)
        #label = torch.full((b_size,), real_label).to(device)
        output_real = netD(real)
        #errD_real = criterion(output, label)
        #errD_real.backward()  # calculate gradients for D
        D_x = output_real.mean().item()
        # Train with fake batches
        noise = torch.randn(b_size, params['nz'], 1, 1).to(device)
        fake = netG(noise)
        #label.fill_(fake_label)
        output_fake = netD(fake.detach())
        #errD_fake = criterion(output, label)
        #errD_fake.backward()  # calculate gradients (gradients accumulated on D)
        #errD = errD_real + errD_fake  # addition in error
        gradient_penalty = compute_gradient_penalty(netD, real.data, fake.detach().data)
        errD = -torch.mean(output_real) + torch.mean(output_fake) + params['lambda_gp'] * gradient_penalty
        errD.backward()
        optimizerD.step()  # update D

        # (2) Update G network: maximize log(D(G(z)))
        if (i+1) % params['n_critic'] == 0:
            #label.fill_(real_label)
            netG.zero_grad()
            output_fake = netD(fake)
            errG = -torch.mean(output_fake)
            errG.backward()  # calculate gradients for G
            D_G_z = output_fake.mean().item()
            optimizerG.step()  # update G

        # Output training stats
        if (i+1) % mini_step == 0:
            time_elapsed = time.time() - since
            time_rest = time_elapsed % 3600
            print('Epoch: [%d/%d]\tStep: [%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f\tTime: %.0fh %.0fm %.0fs'
                  % (epoch, params['num_epochs'], i+1, len(train_data_loader), errD.item(), errG.item(), D_x, D_G_z, time_elapsed // 3600, time_rest // 60, time_rest % 60))
            # Save Losses and Probabilities for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            D_x_set.append(D_x)
            D_G_z_set.append(D_G_z)

    # Check how the generator is doing by saving G's output on fixed_noise
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            # training result with fixed noise
            fake = netG(fixed_noise).detach().cpu()
            plt.figure(figsize=(8, 8))
            plt.axis("off")
            plt.title("Train result with epoch: " + str(epoch+1))
            plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1, 2, 0)))
            plt.savefig('./data/train_result/epoch_' + str(epoch+1) + '.png', bbox_inches='tight')
            plt.close()
            # loss graph
            plt.figure(figsize=(10, 5))
            plt.title("Generator and Discriminator Loss (epoch " + str(epoch+1) + ")")
            plt.plot(G_losses, label="G")
            plt.plot(D_losses, label="D")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig('./data/learning_curve/loss_epoch_' + str(epoch+1) + '.png', bbox_inches='tight')
            plt.close()
            # D_x D_G_z graph
            plt.figure(figsize=(10, 5))
            plt.title("D(x) and D(G(z)) (epoch " + str(epoch+1) + ")")
            plt.plot(D_x_set, label="D(x)")
            plt.plot(D_G_z_set, label="D(G(z))")
            plt.xlabel("iterations")
            plt.ylabel("probability")
            plt.legend()
            plt.savefig('./data/learning_curve/accuracy_epoch_' + str(epoch+1) + '.png', bbox_inches='tight')
            plt.close()
            torch.save(netG.state_dict(), './data/weights/netG_' + str(epoch+1) + '.pth')
            torch.save(netD.state_dict(), './data/weights/netD_' + str(epoch+1) + '.pth')
