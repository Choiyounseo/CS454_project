import torch
import torch.nn as nn
import torchvision.utils as utils
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def imshow_images(rec_rr, zs, netG):
	fig = plt.figure()
	for i in range(rec_rr):
		img = netG(zs[i]).detach()
		img = (img + 1) / 2
		img = img.squeeze()
		np_img = img.numpy()
		ax = fig.add_subplot(1,rec_rr,i+1)
		ax.imshow(np_img)
	plt.show()

def defensegan_gd(x, params, netG, observation_change=False, observation_step=100):
	zs = []
	opts = []

	for i in range(params['r']):
		zs.append(torch.randn((x.shape[0], 100, 1, 1), requires_grad=True, device=device))
		optimizer = torch.optim.SGD([zs[i]], lr=params['lr'], momentum=0.7)
		opts.append(optimizer)
	for l in range(params['L']):
		for i in range(params['r']):
			samples = netG(zs[i])
			MSE_loss = nn.MSELoss()
			loss_mse = MSE_loss(samples[0], x)
			opts[i].zero_grad()
			loss_mse.backward()
			opts[i].step()

		if observation_change and l % observation_step == 0:
			imshow_images(params['r'], zs, netG)

	MSE_loss = nn.MSELoss()
	optimal_loss = MSE_loss(netG(zs[0])[0], x)
	z_hat = zs[0]
	for i in range(1, params['r']):
		MSE_loss = nn.MSELoss()
		loss_mse = MSE_loss(netG(zs[i])[0], x)
		if optimal_loss.ge(loss_mse):
			optimal_loss = loss_mse
			z_hat = zs[i]

	return netG(z_hat)
