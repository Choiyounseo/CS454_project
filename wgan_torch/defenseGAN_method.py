import numpy as np
from torchvision import transforms
import torch
from torch import nn
from model import Generator
import random
from deap import creator, base, tools
from matplotlib import pyplot as plt
from GA import GA
from GD import GD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

'''
def defensegan_memetic_ga(x, params, netG):
	z_array = []
	for i in range(params['r']):
		z_array.append(torch.FloatTensor(params['nz'], 1, 1).normal_(0, 1).numpy())

	result = None
	total_time = 0

	for i in range(int(params['L']/2)):
		z_array, result, time = GA(x, params, netG, z_array)
		total_time += time
		z_array, result, time = GD(x, params, netG, z_array)
		total_time += time

	return result, total_time
'''


def defensegan_ga_gd(x, params, netG):
	z_array = []
	for i in range(params['p']):
		z_array.append(torch.FloatTensor(params['nz'], 1, 1).normal_(0, 1).numpy())

	result = None
	total_time = 0

	for i in range(int(params['L']/2)):
		z_array, result, time = GA(x, params, netG, z_array)
		total_time += time

	for i in range(int(params['L']/2)):
		z_array, result, time = GD(x, params, netG, z_array[:params['r']])
		total_time += time

	return result, total_time

def defensegan_ga(x, params, netG):
	z_array = []
	for i in range(params['p']):
		z_array.append(torch.FloatTensor(params['nz'], 1, 1).normal_(0, 1).numpy())

	result = None
	total_time = 0

	for i in range(params['L']):
		z_array, result, time = GA(x, params, netG, z_array)
		total_time += time

	return result, total_time

def defensegan_gd(x, params, netG):
	z_array = []
	for i in range(params['r']):
		z_array.append(torch.FloatTensor(params['nz'], 1, 1).normal_(0, 1).numpy())

	result = None
	total_time = 0

	for i in range(params['L']):
		z_array, result, time = GD(x, params, netG, z_array)
		total_time += time

	return result, total_time

'''
if __name__ == "__main__":
	netG = Generator(params['ngpu'], params['nc'], params['nz'], params['ngf'])
	netG.load_state_dict(torch.load('./data/weights/netG_12500.pth', map_location=torch.device('cpu')))
	transform = transforms.Compose([transforms.ToTensor()])

	file_path = '../GD/data/fgsm_images/0.3_8_to_5_84.pt'

	fgsm_image = torch.load(file_path)[0]

	defensegan_memetic_ga(fgsm_image, params, netG)
'''
