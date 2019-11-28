import torch
import torch.nn as nn
import torchvision.utils as utils
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from model import Generator
from classifiers.a import ClassifierA
from torch.autograd import Variable
import glob

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
model_weight_path = './data/weights_/netG_12500.pth'
classifier_weight_path = './classifiers/checkpoint'
netG = None

def imshow(img):
	img = (img + 1) / 2
	img = img.squeeze()
	np_img = img.numpy()
	plt.imshow(np_img, cmap='gray')
	plt.show()

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

def defensegan(x, observation_change=False, observation_step=100):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	rr = 10
	L = 200
	lr = 500
	zs = []

	for i in range(rr):
		zs.append(torch.randn((x.shape[0], 100)).view(-1, 100, 1, 1).to(device))
		zs[i].requires_grad = True
	for l in range(L):
		for i in range(rr):
			samples = netG(zs[i])
			MSE_loss = nn.MSELoss()
			loss_mse = MSE_loss(samples[0], x)
			loss_mse.backward()
			zs[i] = zs[i] - lr * zs[i].grad
			zs[i] = zs[i].detach()  # not leaf
			zs[i].requires_grad = True

		if observation_change and l % observation_step == 0:
			imshow_images(rr, zs, netG)

	MSE_loss = nn.MSELoss()
	optimal_loss = MSE_loss(netG(zs[0])[0], x)
	z_hat = zs[0]
	for i in range(1, rr):
		MSE_loss = nn.MSELoss()
		loss_mse = MSE_loss(netG(zs[i])[0], x)
		if optimal_loss.le(loss_mse):
			optimal_loss = loss_mse
			z_hat = zs[i]

	return netG(z_hat)

def main():
	# Generator(ngpu, nc, nz, ngf)
	global netG
	netG = Generator(params['ngpu'], params['nc'], params['nz'], params['ngf'])
	netG.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

	# Classifier
	classifier_a = ClassifierA()
	classifier_a.load_state_dict(torch.load(classifier_weight_path + '_a.pt', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
	classifier_a.eval()

	correct = {}
	total = {}

	for file_path in glob.glob("./data/fgsm_images/*.jpg"):
		# get epsilon and truth by parsing
		epsilon = file_path.split('/')[-1].split('_')[0]
		truth = file_path.split('/')[-1].split('_')[1]

		# image load and convert
		fgsm_image=mpimg.imread(file_path)
		fgsm_image = fgsm_image.reshape(1,fgsm_image.shape[0],fgsm_image.shape[1])
		fgsm_image = torch.from_numpy(fgsm_image).float()

		# do defensegan
		result = defensegan(fgsm_image)

		# todo do classify image
		# output_origin = classifier_a(fgsm_image.unsqueeze(0)).data.max(1, keepdim=True)[1].item()
		output_defense = classifier_a.forward(result).data.max(1, keepdim=True)[1].item()

		if epsilon+'-'+truth in total:
			total[epsilon+'-'+truth] += 1
		else:
			total[epsilon+'-'+truth] = 1

		if epsilon+'-'+truth not in correct:
			correct[epsilon+'-'+truth] = 0
		if output_defense == int(truth):
			correct[epsilon+'-'+truth] += 1

	for k, v in sorted(total.items()):
		print("{} : {}%".format(k, correct[k]/v*100))


if __name__ == "__main__":
	main()