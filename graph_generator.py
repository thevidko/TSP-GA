import numpy as np
import random
import math  # Potřebujeme pro výpočet vzdálenosti


class DistanceMatrix:
    def __init__(self, number_of_cities=10, coord_range_max=1000):
        """
        Inicializuje matici vzdáleností.
        Generuje náhodné souřadnice pro města a počítá Euklidovské vzdálenosti.

        Args:
            number_of_cities (int): Počet měst.
            coord_range_max (int): Maximální hodnota pro X a Y souřadnice měst.
                                   Města budou mít souřadnice v rozsahu [0, coord_range_max].
        """
        if number_of_cities < 2:
            raise ValueError("Počet měst musí být alespoň 2.")

        self.number_of_cities = int(number_of_cities)
        self.coord_range_max = int(coord_range_max)
        self.coordinates = np.zeros((self.number_of_cities, 2))  # Pro uložení (x, y)
        self.distances = np.zeros((self.number_of_cities, self.number_of_cities))
        self._generate_coordinates_and_distances()

    def _generate_coordinates_and_distances(self):
        """Generuje náhodné souřadnice a vypočítá matici vzdáleností."""
        for city in range(self.number_of_cities):
            # Generuj náhodné (x, y) souřadnice
            self.coordinates[city][0] = random.randint(0, self.coord_range_max)
            self.coordinates[city][1] = random.randint(0, self.coord_range_max)

        # Vypočítej vzdálenosti mezi všemi páry měst
        for city in range(self.number_of_cities):
            for to_city in range(city, self.number_of_cities):  # Stačí počítat horní trojúhelník
                if city == to_city:
                    self.distances[city][to_city] = 0
                else:
                    # Euklidovská vzdálenost
                    dist = math.sqrt(
                        (self.coordinates[city][0] - self.coordinates[to_city][0]) ** 2 +
                        (self.coordinates[city][1] - self.coordinates[to_city][1]) ** 2
                    )
                    self.distances[city][to_city] = dist
                    self.distances[to_city][city] = dist  # Matice je symetrická

    def get_distance(self, city1, city2):
        """Vrátí vzdálenost mezi city1 a city2."""
        # Zajistíme, že indexy jsou integer
        return self.distances[int(city1)][int(city2)]

    def get_coordinates(self):
        """Vrátí pole se souřadnicemi měst."""
        return self.coordinates

    def print_matrix(self):
        """Vytiskne matici vzdáleností (zaokrouhleně)."""
        with np.printoptions(precision=2, suppress=True):
            print("Matice vzdáleností:")
            print(self.distances)

    def print_coordinates(self):
        """Vytiskne souřadnice měst."""
        print("\nSouřadnice měst:")
        for i, coord in enumerate(self.coordinates):
            print(f"Město {i}: ({coord[0]:.0f}, {coord[1]:.0f})")


if __name__ == "__main__":
    matrix = DistanceMatrix(number_of_cities=5, coord_range_max=100)
    matrix.print_coordinates()
    matrix.print_matrix()
    print(f"\nVzdálenost mezi městem 0 a 1: {matrix.get_distance(0, 1):.2f}")