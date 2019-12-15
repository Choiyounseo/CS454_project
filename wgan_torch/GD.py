import torch
import torch.nn as nn
import torchvision.utils as utils
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

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

def GD(fgsm_image, params, netG, z_array, print_debug=False):
	optimal_loss = None
	z_hat = None
	zs = []
	opts = []

	for i in range(params['r']):
		zs.append(torch.tensor(z_array[i].reshape(1, params['nz'], 1, 1), requires_grad=True, device=device))
		optimizer = torch.optim.SGD([zs[i]], lr=params['lr'], momentum=0.7)
		opts.append(optimizer)

	start = time.time()

	for i in range(params['r']):
		samples = netG(zs[i])
		MSE_loss = nn.MSELoss()
		loss_mse = MSE_loss(samples[0], fgsm_image)
		opts[i].zero_grad()
		loss_mse.backward()
		opts[i].step()
		if optimal_loss is None or optimal_loss.ge(loss_mse):
			optimal_loss = loss_mse
			z_hat = zs[i]

	end = time.time()

	if print_debug:
		imshow_images(params['r'], zs, netG)

	# print(optimal_loss)

	return [z.detach().numpy().reshape(params['nz'], 1, 1) for z in zs], netG(z_hat), end-start, optimal_loss
