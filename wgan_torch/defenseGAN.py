import torch
import torch.nn as nn
import torchvision.utils as utils
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from model import Generator
from classifiers.a import ClassifierA
from classifiers.b import ClassifierB
from classifiers.c import ClassifierC
import glob
from torchvision import transforms
from defenseGAN_method import defensegan_ga_gd, defensegan_ga, defensegan_gd

# Hyper parameters
params = {
	'input_size': 28,  # image size 1x64x64
	'p': 50,   # population size for GA
	'r': 10,   # population size for GD
	'L': 200,  # number of iterations
	'lr': 10,  # learning rate
	'nc': 1,  # number of channels
	'nz': 100,  # size of z latent vector
	'ngf': 64,  # size of feature maps in generator
	'ngpu': 1,  # number of GPU
}

model_weight_path = './data/weights/netG_12500.pth'
classifier_weight_path = './classifiers/checkpoint'
classifier_model_version = 'A'

fgsm_image_path = './data/classifier_a_fgsm_tensors/*.pt'
#fgsm_image_path = './data/classifier_b_fgsm_small_tensors/*.pt'
#fgsm_image_path = './data/classifier_c_fgsm_sample/*.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
netG = None

# transform
transform = transforms.Compose([transforms.Normalize(mean=(0.5,), std=(0.5,))])

def imshow(img):
	img = (img + 1) / 2
	img = img.squeeze()
	np_img = img.numpy()
	plt.imshow(np_img, cmap='gray')
	plt.show()

def load_classifier(version):
	classifier = None
	if (version == 'A'):
		classifier = ClassifierA()
		classifier.load_state_dict(torch.load(classifier_weight_path + '_a.pt', map_location=device))
	elif (version == 'B'):
		classifier = ClassifierB()
		classifier.load_state_dict(torch.load(classifier_weight_path + '_b.pt', map_location=device)['model'])
	elif (version == 'C'):
		classifier = ClassifierC()
		classifier.load_state_dict(torch.load(classifier_weight_path + '_c.pt', map_location=device)['model'])

	classifier.eval()
	return classifier

def main():
	# Generator(ngpu, nc, nz, ngf)
	global netG
	netG = Generator(params['ngpu'], params['nc'], params['nz'], params['ngf'])
	netG.load_state_dict(torch.load(model_weight_path, map_location=device))

	# Classifier
	classifier = load_classifier(classifier_model_version)

	# available epsilon values : 0.1, 0.2, 0.3, 0.05, 0.15, 0.25
	epsilon_set = [0.1, 0.2, 0.3, 0.05, 0.15, 0.25]
	acc_defense_gan = [0] * 6  # accuracy of defense GAN
	acc_classifier = [0] * 6  # accuracy of classifier a
	total = [0] * 6  # number of fgsm images for each epsilon
	correct_defense_gan = [0] * 6  # number of fgsm images correctly classified for each epsilon by defense gan
	correct_classifier = [0] * 6  # number of fgsm images correctly classified for each epsilon by classifier a

	total_time = 0

	i = 1
	file_paths = glob.glob(fgsm_image_path)
	file_paths.sort()
	for file_path in file_paths:  # fgsm images from classifier a (fgsm_images_a)
		print('file number : ' + str(i))
		i += 1

		# get epsilon and ground truth by parsing
		file_name = file_path.split('\\')[-1].split('_')
		print(file_name)
		epsilon = float(file_name[0])
		ground_truth = float(file_name[1])
		fgsm_truth = float(file_name[3])
		print('This fgsm image is originally ' + str(int(ground_truth)) + ', misclassified as ' +
			  str(int(fgsm_truth)) + ' with epsilon ' + str(epsilon))
		fgsm_image = torch.load(file_path)[0]
		# imshow(fgsm_image)

		''' YOU CAN PICK ONE OF THOSE METHODS BELOW. THOSE ARE FROM defenseGAN_method.py
		1. defensegan_memetic_ga : 100 GA + 100 GD (alternatively)
		2. defensegan_ga_ga      : 100 GA + 100 GD (in order)
		3. defensegan_ga         : 200 GA
		4. defensegan_gd         : 200 GD
		'''
		#result_image, _total_time = defensegan_memetic_ga(fgsm_image, params, netG)
		result_image, _total_time = defensegan_ga_gd(fgsm_image, params, netG)
		#result_image, _total_time = defensegan_ga(fgsm_image, params, netG)
		#result_image, _total_time = defensegan_gd(fgsm_image, params, netG)

		total_time += _total_time

		# to classify image
		outputs_defense_gan = classifier(result_image)
		outputs_classifier = classifier(fgsm_image.view(1, 1, params['input_size'], params['input_size']))
		prediction_defense_gan = torch.max(outputs_defense_gan.data, 1)[1]
		prediction_classifier = torch.max(outputs_classifier.data, 1)[1]

		print('defense gan classified fgsm image (' + str(ground_truth) + ' to ' + str(fgsm_truth) +
			  ') as ' + str(prediction_defense_gan.item()))
		print('classifier classified fgsm image (' + str(ground_truth) + ' to ' + str(fgsm_truth) +
			  ') as ' + str(prediction_classifier.item()))
		epsilon_index = epsilon_set.index(epsilon)  # returns the index of epsilon from epsilon_set
		total[epsilon_index] += 1

		if prediction_defense_gan.item() == ground_truth:
			print('prediction from defense gan correct!')
			correct_defense_gan[epsilon_index] += 1
		if prediction_classifier.item() == ground_truth:
			print('prediction from classifier correct! - this should not happen...')
			correct_classifier[epsilon_index] += 1
		print()
		print('total # images for each epsilon : ' + str(total))
		print('correct defense gan : ' + str(correct_defense_gan))
		print('correct classifier : ' + str(correct_classifier))
		break
	time_rest = total_time % 3600
	print('total time for this method : %.0fh %.0fm %.0fs' % (total_time // 3600, time_rest // 60, time_rest % 60))

if __name__ == "__main__":
	main()