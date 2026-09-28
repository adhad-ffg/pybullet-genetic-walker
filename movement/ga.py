import numpy as np
import config
from network import NeuralNetwork

class GeneticAlgorithm:
    def __init__(self):
        self.param_count = NeuralNetwork().param_count

    def create_initial_population(self):
        return [NeuralNetwork() for _ in range(config.POPULATION_SIZE)]

    def select_parents(self, population, fitnesses):
        sorted_indices = np.argsort(fitnesses)[::-1]
        elites = [population[i] for i in sorted_indices[:config.ELITE_COUNT]]
        
        min_fit = np.min(fitnesses)
        adjusted_fit = fitnesses - min_fit + 1e-6
        probs = adjusted_fit / np.sum(adjusted_fit)
        
        parents = []
        for _ in range(config.POPULATION_SIZE - config.ELITE_COUNT):
            parent = np.random.choice(population, p=probs)
            parents.append(parent)
        return elites, parents

    def crossover(self, parent1, parent2):
        mask = np.random.rand(self.param_count) > 0.5
        child_weights = np.where(mask, parent1.weights, parent2.weights)
        return NeuralNetwork(child_weights)

    def mutate(self, individual):
        mutation_mask = np.random.rand(self.param_count) < config.MUTATION_RATE
        noise = np.random.normal(0.0, config.MUTATION_STDEV, self.param_count)
        individual.weights += mutation_mask * noise
        individual.__init__(individual.weights)

    def evolve(self, population, fitnesses):
        elites, parents = self.select_parents(population, fitnesses)
        next_population = list(elites)
        
        while len(next_population) < config.POPULATION_SIZE:
            p1, p2 = np.random.choice(parents, 2, replace=False)
            child = self.crossover(p1, p2)
            self.mutate(child)
            next_population.append(child)
            
        return next_population
