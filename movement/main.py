import time
import numpy as np
import config
from environment import WalkerEnvironment
from ga import GeneticAlgorithm

def main():
    env = WalkerEnvironment()
    ga = GeneticAlgorithm()
    population = ga.create_initial_population()

    for gen in range(config.MAX_GENERATIONS):
        fitnesses = []
        
        for idx, individual in enumerate(population):
            obs = env.reset()
            total_fitness = 0
            
            for _ in range(config.SIM_STEPS_PER_INDIVIDUAL):
                action = individual.forward(obs)
                env.apply_action(action)
                env.step()
                obs = env.get_observation()
                
                if config.RENDER_SIMULATION:
                    time.sleep(config.TIME_STEP)
            
            fitness = env.get_fitness()
            fitnesses.append(fitness)
            
        fitnesses = np.array(fitnesses)
        print(f"Gen {gen} | Max Fit: {np.max(fitnesses):.4f} | Avg Fit: {np.mean(fitnesses):.4f}")
        
        population = ga.evolve(population, fitnesses)

    env.close()

if __name__ == "__main__":
    main()
