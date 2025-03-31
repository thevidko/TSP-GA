import numpy as np
import random

# TODO: implementace nespojených měst
class DistanceMatrix:
    def __init__(self, number_of_cities=100, min_distance=50, max_distance=2000):
        self.number_of_cities = number_of_cities
        self.distances = np.zeros((number_of_cities, number_of_cities))
        self.min_distance = min_distance
        self.max_distance = max_distance
        self._generate_distances()

    def _generate_distances(self):
        for city in range(self.number_of_cities):
            for to_city in range(self.number_of_cities):
                if city != to_city:
                    self.distances[city][to_city] = self.distances[to_city][city] = random.randint(self.min_distance, self.max_distance)

    def get_distance(self, city1, city2):
        return self.distances[city1][city2]

    def print_matrix(self):
        print(self.distances)

if __name__ == "__main__":
    matrix = DistanceMatrix(number_of_cities=5, min_distance=50, max_distance=200)
    matrix.print_matrix()
    print("Vzdálenost mezi městem 0 a 1:", matrix.get_distance(0, 1))