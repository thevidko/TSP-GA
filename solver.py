import numpy as np
import random
import itertools


class TSPGeneticSolver:
    def __init__(self, distance_matrix, num_generations=200, population_size=100, mutation_rate=0.1):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def fitness(self, route):
        """ Výpočet fitness (1/celková vzdálenost) """
        dist = sum(self.distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
        dist += self.distance_matrix[route[-1]][route[0]]  # Návrat do startu
        return 1 / dist if dist else float('inf')

    def mutate(self, route):
        """ Swap mutace - prohození dvou náhodných měst """
        a, b = random.sample(range(len(route)), 2)
        route[a], route[b] = route[b], route[a]
        return route

    def crossover(self, parent1, parent2):
        """ PMX crossover - vezmeme část jednoho rodiče a doplníme chybějící města """
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))
        child = [-1] * size
        child[a:b] = parent1[a:b]
        pos = b
        for i in itertools.chain(range(b, size), range(0, b)):
            if parent2[i] not in child:
                child[pos] = parent2[i]
                pos = (pos + 1) % size
        return child

    def solve(self):
        """ Spuštění genetického algoritmu """
        if self.num_cities < 2:
            return []

        # Počáteční populace (náhodné permutace)
        population = [random.sample(range(self.num_cities), self.num_cities) for _ in range(self.population_size)]

        for _ in range(self.num_generations):
            # Ohodnocení fitness funkcí
            scored_population = sorted(population, key=lambda route: -self.fitness(route))

            # Selekce (elitismus + turnaj)
            next_gen = scored_population[:10]  # Nejlepší přežijí
            for _ in range(self.population_size - 10):
                parent1, parent2 = random.choices(scored_population[:50], k=2)
                child = self.crossover(parent1, parent2)
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                next_gen.append(child)

            population = next_gen

        # Nejlepší nalezená cesta
        return scored_population[0]
