import torch
import torch.nn as nn
import torchvision.utils as utils
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from model import Generator
from classifiers.a import ClassifierA
from classifiers.b import ClassifierB
from torch.autograd import Variable
import glob
from torchvision import datasets, transforms
from PIL import Image
import torchvision.transforms.functional as TF
import random
from deap import creator, base, tools, algorithms
import os

# Hyper parameters
params = {
	'input_size': 28,  # image size 1x64x64
	'r': 10,   # population size
	'L': 200,  # number of iterations
	'lr': 500,  # learning rate
	'nc': 1,  # number of channels
	'nz': 100,  # size of z latent vector
	'ngf': 64,  # size of feature maps in generator
	'ngpu': 1,  # number of GPU
}

model_weight_path = './data/weights/netG_12500.pth'
classifier_weight_path = './classifiers/checkpoint'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netG = None

# transform
transform = transforms.Compose([transforms.Normalize(mean=(0.5,), std=(0.5,))])

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


def defensegan_gd(x, observation_change=False, observation_step=100):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	zs = []

	for i in range(params['r']):
		zs.append(torch.randn((x.shape[0], 100)).view(-1, 100, 1, 1).to(device))
		zs[i].requires_grad = True
	for l in range(params['L']):
		for i in range(params['r']):
			samples = netG(zs[i])
			MSE_loss = nn.MSELoss()
			loss_mse = MSE_loss(samples[0], x)
			loss_mse.backward()
			zs[i] = zs[i] - params['lr'] * zs[i].grad
			zs[i] = zs[i].detach()  # not leaf
			zs[i].requires_grad = True

		if observation_change and l % observation_step == 0:
			imshow_images(params['r'], zs, netG)

	MSE_loss = nn.MSELoss()
	optimal_loss = MSE_loss(netG(zs[0])[0], x)
	z_hat = zs[0]
	for i in range(1, params['r']):
		MSE_loss = nn.MSELoss()
		loss_mse = MSE_loss(netG(zs[i])[0], x)
		if optimal_loss.le(loss_mse):
			optimal_loss = loss_mse
			z_hat = zs[i]

	return netG(z_hat)

def defensegan_ga(x, observation_change=False, observation_step=100):
	x = x.view(28, 28).detach().numpy().astype(np.float64)
	initial_population = torch.FloatTensor(params['r'], params['nz'], 1, 1).normal_(0, 1)
	initial_population = initial_population.view(params['r'], params['nz']).numpy()
	def evalFunc(individual):
		individual = torch.from_numpy(individual).view(1, params['nz'], 1, 1)
		fitness = np.linalg.norm(netG(individual).view(28, 28).detach().numpy() - x, ord=2) ** 2,
		return fitness
	def initIndividual(icls, content):
		return icls(content)
	def initPopulation(pcls, ind_init):
		return pcls(ind_init(c) for c in initial_population)
	creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
	creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)  # minimizing the fitness value
	toolbox = base.Toolbox()
	IND_SIZE = params['nz']
	POPULATION = params['r']
	CXPB, MUTPB = 0.4, 0.2
	GENERATIONS = params['L']
	toolbox.register("attr_float", random.random)
	toolbox.register("individual", initIndividual, creator.Individual)
	toolbox.register("population", initPopulation, list, toolbox.individual)
	toolbox.register("evaluate", evalFunc)
	toolbox.register("mate", tools.cxUniform, indpb=0.1)
	toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
	toolbox.register("select", tools.selTournament, tournsize=3)

	random.seed(777)

	# pop = toolbox.population(n=POPULATION)
	pop = toolbox.population()

	print("Start of evolution")

	# Evaluate the entire population
	# print(fitnesses) -> [(84,), (105,), (96,), (104,), (94,),  ... ] 이런식으로 저장됨.
	fitnesses = list(map(toolbox.evaluate, pop))
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit

	# Extracting all the fitnesses of
	fits = [ind.fitness.values[0] for ind in pop]

	# Variable keeping track of the number of generations
	g = 0

	# Begin the evolution
	while min(fits) > 10 and g < GENERATIONS:
		# A new generation
		g = g + 1

		# Select the next generation individuals
		# len(pop) -> 50, len(pop[0]) -> 5
		offspring = toolbox.select(pop, len(pop))

		# Clone the selected individuals
		offspring = list(map(toolbox.clone, offspring))

		# Apply crossover and mutation on the offspring
		'''
        they modify those individuals within the toolbox container
        and we do not need to reassign their results.
        '''
		# TODO: want p_new1 = p_m - beta(p_m - p_d), p_new2 = p_m + beta(p_m - p_d)
		# want to customize mutation method... there is no proper mutation operator in deap.tools...

		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			if random.random() < CXPB:
				size = min(len(child1), len(child2))
				for i in range(5):
					cxpoint = random.randint(2, size - 1)
					mtpoint = cxpoint - 1
					# cxpoint -1 위치 : mutate
					beta = random.random()
					child1[mtpoint] = child1[mtpoint] - beta * (child1[mtpoint] - child2[mtpoint])
					child2[mtpoint] = child1[mtpoint] + beta * (child1[mtpoint] - child2[mtpoint])

				# crossover : one point crossover (temporary crossover algorithm)
				# child1[cxpoint:], child2[cxpoint:] = child2[cxpoint:], child1[cxpoint:]
				del child1.fitness.values
				del child2.fitness.values

		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			if random.random() < CXPB:
				toolbox.mate(child1, child2)
				del child1.fitness.values
				del child2.fitness.values

		for mutant in offspring:
			if random.random() < MUTPB:
				toolbox.mutate(mutant)
				del mutant.fitness.values

		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		# The population is entirely replaced by the offspring
		pop[:] = offspring

		# Gather all the fitnesses in one list and print the stats
		fits = [ind.fitness.values[0] for ind in pop]

		length = len(pop)
		mean = sum(fits) / length
		sum2 = sum(x * x for x in fits)
		std = abs(sum2 / length - mean ** 2) ** 0.5

		if g % 50 == 0:
			print("-- Generation %i --" % g)
			print("  Min %s" % min(fits))
			print("  Max %s" % max(fits))
			print("  Avg %s" % mean)
			print("  Std %s" % std)
			best_ind = tools.selBest(pop, 1)[0]
			z = torch.from_numpy(best_ind).view(1, 100, 1, 1)
			gen_image = netG(z)
			# imshow(gen_image.detach())

	print("-- End of (successful) evolution --")

	best_ind = tools.selBest(pop, 1)[0]
	z = torch.from_numpy(best_ind).view(1, 100, 1, 1)
	gen_image = netG(z)
	# imshow(gen_image.detach())
	return gen_image

def main():
	# Generator(ngpu, nc, nz, ngf)
	global netG
	netG = Generator(params['ngpu'], params['nc'], params['nz'], params['ngf'])
	netG.load_state_dict(torch.load(model_weight_path, map_location=device))

	# Classifier
	classifier_a = ClassifierA()
	# classifier_b.load_state_dict(torch.load(classifier_weight_path + '_b.pt', map_location=device)['model'])
	classifier_a.load_state_dict(torch.load(classifier_weight_path + '_a.pt', map_location=device))
	classifier_a.eval()

	# available epsilon values : 0.1, 0.2, 0.3, 0.05, 0.15, 0.25
	epsilon_set = [0.1, 0.2, 0.3, 0.05, 0.15, 0.25]
	acc_defense_gan = [0] * 6  # accuracy of defense GAN
	acc_classifier_a = [0] * 6  # accuracy of classifier a
	total = [0] * 6  # number of fgsm images for each epsilon
	correct_defense_gan = [0] * 6  # number of fgsm images correctly classified for each epsilon by defense gan
	correct_classifier_a = [0] * 6  # number of fgsm images correctly classified for each epsilon by classifier a

	for file_path in glob.glob("./data/classifier_a_fgsm_small_tensors/*.pt"):  # fgsm images from classifier a (fgsm_images_a)
		# get epsilon and ground truth by parsing
		file_name = file_path.split('/')[-1].split('_')
		epsilon = float(file_name[0])
		ground_truth = float(file_name[1])
		fgsm_truth = float(file_name[3])
		print('This fgsm image is originally ' + str(int(ground_truth)) + ', misclassified as ' +
			  str(int(fgsm_truth)) + ' with epsilon ' + str(epsilon))
		fgsm_image = torch.load(file_path)[0]
		# imshow(fgsm_image)
		# do defense gan
		result_image = defensegan_gd(fgsm_image)  # return type tensor [1, 1, 28, 28]. image G(z) that has minimum fitness
		# to classify image
		outputs_defense_gan = classifier_a(result_image)
		outputs_classifier_a = classifier_a(fgsm_image.view(1, 1, params['input_size'], params['input_size']))
		prediction_defense_gan = torch.max(outputs_defense_gan.data, 1)[1]
		prediction_classifier_a = torch.max(outputs_classifier_a.data, 1)[1]
		#print('output from defense gan is ' + str(outputs_defense_gan))
		#print('output from classifier is ' + str(outputs_classifier_a))
		print('defense gan classified fgsm image (' + str(ground_truth) + ' to ' + str(fgsm_truth) +
			  ') as ' + str(prediction_defense_gan.item()))
		print('classifier a classified fgsm image (' + str(ground_truth) + ' to ' + str(fgsm_truth) +
			  ') as ' + str(prediction_classifier_a.item()))
		epsilon_index = epsilon_set.index(epsilon)  # returns the index of epsilon from epsilon_set
		total[epsilon_index] += 1
		if prediction_defense_gan.item() == ground_truth:
			print('prediction from defense gan correct!')
			correct_defense_gan[epsilon_index] += 1
		if prediction_classifier_a.item() == ground_truth:
			print('prediction from classifier correct! - this should not happen...')
			correct_classifier_a[epsilon_index] += 1
		print()
		# break
	print('total # images for each epsilon : ' + str(total))
	print('correct defense gan : ' + str(correct_defense_gan))
	print('correct classifier a : ' + str(correct_classifier_a))

if __name__ == "__main__":
	main()