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
	'r': 10,   # population size
	'L': 200,  # number of iterations
	'lr': 500,  # learning rate
	'nc': 1,  # number of channels
	'nz': 100,  # size of z latent vector
	'ngf': 64,  # size of feature maps in generator
	'ngpu': 1,  # number of GPU
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

	for file_path in glob.glob("./data/classifier_a_fgsm_small_tensors/*.pt"):
		print(file_path)
		# get epsilon and truth by parsing
		epsilon = file_path.split('/')[-1].split('_')[0]
		ground_truth = file_path.split('/')[-1].split('_')[1]
		fgsm_truth = file_path.split('/')[-1].split('_')[3]

		# image load and convert
		fgsm_image = torch.load(file_path)[0]

		# do defensegan
		result = defensegan(fgsm_image)

		# todo do classify image
		# output_origin = classifier_a(fgsm_image.unsqueeze(0)).data.max(1, keepdim=True)[1].item()
		output_defense = classifier_a.forward(result).data.max(1, keepdim=True)[1].item()

		if epsilon+'-'+ground_truth in total:
			total[epsilon+'-'+ground_truth] += 1
		else:
			total[epsilon+'-'+ground_truth] = 1

		if epsilon+'-'+ground_truth not in correct:
			correct[epsilon+'-'+ground_truth] = 0
		if output_defense == int(ground_truth):
			correct[epsilon+'-'+ground_truth] += 1
		print("epsilon: {}, groud: {}, fgsm: {}, defansegan: {}".format(epsilon, ground_truth, fgsm_truth, output_defense))

	for k, v in sorted(total.items()):
		print("{} : {}%".format(k, correct[k]/v*100))


if __name__ == "__main__":
	main()