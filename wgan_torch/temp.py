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
transform = transforms.Compose([transforms.ToTensor()])
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
netG = Generator(params['ngpu'], params['nc'], params['nz'], params['ngf'])
''' this part can be used later when ngpu > 1
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
'''
netG.apply(weights_init)
print(netG)

# Discriminator(ngpu, nc, ndf)
netD = Discriminator(params['ngpu'], params['nc'], params['ndf'])
netD.apply(weights_init)
print(netD)


input = torch.FloatTensor(params['batch_size'], 1, params['input_size'], params['input_size'])
noise = torch.FloatTensor(params['batch_size'], params['nz'], 1, 1)
fixed_noise = torch.FloatTensor(params['batch_size'], params['nz'], 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1
# CUDA
netD.cuda()
netG.cuda()
input = input.cuda()
one, mone = one.cuda(), mone.cuda()
noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# Criterion & Optimizer
optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))


G_losses = []
D_losses = []
D_x_set = []
D_G_z_set = []
gen_iterations = 0
for epoch in range(params['num_epochs']):
    data_iter = iter(train_data_loader)
    i = 0
    while i < len(train_data_loader):
        ###########################
        # (1) Update D network
        ###########################
        for p in netD.parameters():
            p.requires_grad = True
        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = 5
        j = 0
        while j < Diters and i < len(train_data_loader):
            j += 1

            # clamp parameters to a cube
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)

            data = data_iter.next()
            i += 1

            # train with real
            real_cpu, _ = data
            netD.zero_grad()
            real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            inputv = Variable(input)

            errD_real = netD(inputv)
            errD_real = errD_real.mean(0)
            errD_real = errD_real.view(1)
            errD_real.backward(one)

            # train with fake
            noise.resize_(params['batch_size'], params['nz'], 1, 1).normal_(0, 1)
            noisev = Variable(noise)  # totally freeze netG
            fake = Variable(netG(noisev.detach()).data)
            inputv = fake
            errD_fake = netD(inputv)
            errD_fake = errD_fake.mean(0)
            errD_fake = errD_fake.view(1)
            errD_fake.backward(mone)
            errD = errD_real - errD_fake
            optimizerD.step()
        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noise.resize_(params['batch_size'], params['nz'], 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        errG = netD(fake)
        errG = errG.mean(0)
        errG = errG.view(1)
        errG.backward(one)
        optimizerG.step()
        gen_iterations += 1

        print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
              % (epoch, params['num_epochs'], i, len(train_data_loader), gen_iterations,
                 errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
        G_losses.append(errG.data[0])
        D_losses.append(errD.data[0])
        D_x_set.append(errD_real.data[0])
        D_G_z_set.append(errD_fake.data[0])
        if gen_iterations % 500 == 0:
            with torch.no_grad():
                real_cpu = real_cpu.mul(0.5).add(0.5)
                vutils.save_image(real_cpu, '{0}/real_samples.png'.format('./data/train_result'))
                fake = netG(Variable(fixed_noise))
                fake.data = fake.data.mul(0.5).add(0.5)
                vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format('./data/train_result', gen_iterations))
                # loss graph
                plt.figure(figsize=(10, 5))
                plt.title("Generator and Discriminator Loss (epoch " + str(gen_iterations) + ")")
                plt.plot(G_losses, label="G")
                plt.plot(D_losses, label="D")
                plt.xlabel("iterations")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig('./data/learning_curve/loss_epoch_' + str(gen_iterations) + '.png', bbox_inches='tight')
                plt.close()
                # D_x D_G_z graph
                plt.figure(figsize=(10, 5))
                plt.title("D(x) and D(G(z)) (epoch " + str(gen_iterations) + ")")
                plt.plot(D_x_set, label="D(x)")
                plt.plot(D_G_z_set, label="D(G(z))")
                plt.xlabel("iterations")
                plt.ylabel("probability")
                plt.legend()
                plt.savefig('./data/learning_curve/accuracy_epoch_' + str(gen_iterations) + '.png', bbox_inches='tight')
                plt.close()
                torch.save(netG.state_dict(), './data/weights/netG_' + str(gen_iterations) + '.pth')
                torch.save(netD.state_dict(), './data/weights/netD_' + str(gen_iterations) + '.pth')
