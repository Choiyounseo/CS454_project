import numpy as np
from torchvision import transforms
import torch
from torch import nn
from model import Generator
import random
from deap import creator, base, tools
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper parameters
params = {
	'input_size': 28,  # image size 1x64x64
	'nc': 1,  # number of channels
	'nz': 100,  # size of z latent vector
	'ngf': 64,  # size of feature maps in generator
	'r': 10,   # population size
	'L': 100,  # number of iterations
	'lr': 500,  # learning rate
	'ngpu': 1,  # number of GPU
}

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

def defensegan_gd(z_array, print_debug=False):
	optimal_loss = None
	z_hat = None
	zs = []
	opts = []

	for i in range(params['r']):
		zs.append(torch.tensor(z_array[i].reshape(1, params['nz'], 1, 1), requires_grad=True, device=device))
		optimizer = torch.optim.SGD([zs[i]], lr=params['lr'], momentum=0.7)
		opts.append(optimizer)
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

	if print_debug:
		imshow_images(params['r'], zs, netG)

	print(optimal_loss)

	return [z.detach().numpy().reshape(params['nz'], 1, 1) for z in zs], netG(z_hat)

def defensegan_ga(z_array):
	initial_population = torch.tensor(np.asarray(z_array), device=device)
	initial_population = initial_population.view(params['r'], params['nz']).numpy()
	def evalFunc(individual):
		individual = torch.from_numpy(individual).view(1, params['nz'], 1, 1)
		fitness = np.linalg.norm(netG(individual).view(28, 28).detach().numpy() - fgsm_image.view(28, 28).detach().numpy(), ord=2) ** 2,
		return fitness
	def initIndividual(icls, content):
		return icls(content)
	def initPopulation(pcls, ind_init):
		return pcls(ind_init(c) for c in initial_population)
	creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
	creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)  # minimizing the fitness value
	toolbox = base.Toolbox()
	CXPB, MUTPB = 0.95, 0.05
	toolbox.register("attr_float", random.random)
	toolbox.register("individual", initIndividual, creator.Individual)
	toolbox.register("population", initPopulation, list, toolbox.individual)
	toolbox.register("evaluate", evalFunc)
	toolbox.register("mate", tools.cxUniform, indpb=0.1)
	toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
	toolbox.register("select", tools.selRoulette)

	random.seed(777)

	# pop = toolbox.population(n=POPULATION)
	pop = toolbox.population()

	# print("Start of evolution")

	# Evaluate the entire population
	# print(fitnesses) -> [(84,), (105,), (96,), (104,), (94,),  ... ] 이런식으로 저장됨.
	fitnesses = list(map(toolbox.evaluate, pop))
	minfit = 1000000.0
	elit = None
	for ind, fit in zip(pop, fitnesses):
		if fit[0] < minfit:
			minfit = fit[0]
			elit = ind
		ind.fitness.values = fit

	# Extracting all the fitnesses of
	fits = [ind.fitness.values[0] for ind in pop]


	# Select the next generation individuals
	# len(pop) -> 50, len(pop[0]) -> 5
	offspring = toolbox.select(pop, len(pop)-1)

	# Clone the selected individuals
	offspring = [elit] + list(map(toolbox.clone, offspring))

	# Apply crossover and mutation on the offspring
	'''
    they modify those individuals within the toolbox container
    and we do not need to reassign their results.
    '''
	# TODO: gaussian mutation maybe better..
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
			toolbox.mate(child1, child2)
			toolbox.mate(child1, child2)
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
	print("mean:{}, std:{}\n".format(mean, std))

	return [z for z in pop]

if __name__ == "__main__":
	netG = Generator(params['ngpu'], params['nc'], params['nz'], params['ngf'])
	netG.load_state_dict(torch.load('./data/weights/netG_12500.pth', map_location=torch.device('cpu')))
	transform = transforms.Compose([transforms.ToTensor()])

	file_path = '../GD/data/fgsm_images/0.3_8_to_5_84.pt'

	fgsm_image = torch.load(file_path)[0]

	z_array = []
	for i in range(params['r']):
		z_array.append(torch.FloatTensor(params['nz'], 1, 1).normal_(0, 1).numpy())

	for i in range(params['L']):
		z_array = defensegan_ga(z_array)
		z_array, _ = defensegan_gd(z_array, i%20==0)