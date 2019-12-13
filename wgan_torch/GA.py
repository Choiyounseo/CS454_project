import torch
import torch.nn as nn
import torchvision.utils as utils
import random
import numpy as np
from deap import creator, base, tools, algorithms
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def GA(fgsm_image, params, netG, z_array):
	initial_population = torch.tensor(np.asarray(z_array), device=device)
	initial_population = initial_population.view(params['p'], params['nz']).numpy()
	def evalFunc(individual):
		individual = torch.from_numpy(individual).view(1, params['nz'], 1, 1)
		fitness = np.linalg.norm(netG(individual).view(28, 28).detach().numpy() - fgsm_image.view(28, 28).detach().numpy(), ord=2) ** 2,
		#print(fitness)
		return fitness
	def initIndividual(icls, content):
		return icls(content)
	def initPopulation(pcls, ind_init):
		return pcls(ind_init(c) for c in initial_population)
	creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
	creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)  # minimizing the fitness value
	toolbox = base.Toolbox()
	CXPB, MUTPB = 0.4, 0.2
	toolbox.register("attr_float", random.random)
	toolbox.register("individual", initIndividual, creator.Individual)
	toolbox.register("population", initPopulation, list, toolbox.individual)
	toolbox.register("evaluate", evalFunc)
	toolbox.register("mate", tools.cxUniform, indpb=0.1)
	toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
	toolbox.register("select", tools.selTournament, tournsize=3)
	#toolbox.register("select", tools.selRoulette)

	random.seed(777)

	# pop = toolbox.population(n=POPULATION)
	pop = toolbox.population()

	# print("Start of evolution")

	start = time.time()
	# Evaluate the entire population
	# print(fitnesses) -> [(84,), (105,), (96,), (104,), (94,),  ... ] 이런식으로 저장됨.
	fitnesses = list(map(toolbox.evaluate, pop))
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit

	# Extracting all the fitnesses of
	fits = [ind.fitness.values[0] for ind in pop]

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

	end = time.time()

	# Gather all the fitnesses in one list and print the stats
	fits = [ind.fitness.values[0] for ind in pop]

	length = len(pop)
	mean = sum(fits) / length
	sum2 = sum(x * x for x in fits)
	std = abs(sum2 / length - mean ** 2) ** 0.5
	# print("mean:{}, std:{}\n".format(mean, std))
	best_ind = tools.selBest(pop, 1)[0]
	z = torch.from_numpy(best_ind).view(1, 100, 1, 1)
	# imshow(gen_image.detach())
	return [z for z in pop], netG(z), end - start