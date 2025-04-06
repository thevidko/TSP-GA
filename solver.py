import random
import numpy as np

# TODO: implementace nespojených měst
class TSPGeneticSolver:
    """
    Třída implementující genetický algoritmus pro řešení problému obchodního cestujícího (TSP).
    """

    def __init__(self, distance_matrix, population_size, mutation_rate, n_generations, elite_size=1, tournament_size=3):
        """
        Inicializuje genetický algoritmus.

        Args:
            distance_matrix (DistanceMatrix): Objekt obsahující matici vzdáleností mezi městy.
            population_size (int): Počet jedinců (cest) v populaci.
            mutation_rate (float): Pravděpodobnost, s jakou dojde k mutaci jedince.
            n_generations (int): Počet generací, po které algoritmus poběží.
            elite_size (int): Počet nejlepších jedinců, kteří automaticky postoupí do další generace.
            tournament_size (int): Počet jedinců vybíraných do turnaje při selekci.
        """
        self.distance_matrix = distance_matrix
        self.n_cities = distance_matrix.number_of_cities
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.elite_size = elite_size
        self.tournament_size = tournament_size

        if self.elite_size >= self.population_size:
            raise ValueError("Velikost elity (elite_size) musí být menší než velikost populace (population_size).")
        if self.n_cities <= 1:
            raise ValueError("Počet měst musí být alespoň 2 pro řešení TSP.")

    def _create_individual(self):
        """Vytvoří jednoho jedince (náhodnou cestu)."""
        individual = list(range(self.n_cities))
        random.shuffle(individual)
        return individual

    def _initialize_population(self):
        """Inicializuje populaci náhodnými jedinci."""
        return [self._create_individual() for _ in range(self.population_size)]

    def _calculate_total_distance(self, route):
        """Vypočítá celkovou délku dané cesty."""
        total_distance = 0
        for i in range(self.n_cities):
            from_city = route[i]
            # Použití modula pro zajištění návratu do startovního města
            to_city = route[(i + 1) % self.n_cities]
            total_distance += self.distance_matrix.get_distance(from_city, to_city)
        return total_distance

    def _calculate_fitness(self, route):
        """
        Vypočítá fitness jedince (cesty). Fitness je převrácená hodnota
        celkové vzdálenosti (kratší vzdálenost = vyšší fitness).
        Přidáváme malou hodnotu epsilon, abychom zabránili dělení nulou.
        """
        distance = self._calculate_total_distance(route)
        # Můžeme použít i jiné škálování, ale 1/distance je běžné
        return 1.0 / (distance + 1e-9)

    def _tournament_selection(self, population, fitnesses):
        """
        Provede selekci rodiče pomocí turnajové metody.
        Náhodně vybere 'tournament_size' jedinců a vrátí toho nejlepšího z nich.
        """
        # Vyber náhodné indexy pro turnaj (s opakováním)
        tournament_indices = np.random.randint(0, len(population), self.tournament_size)

        best_idx_in_tournament = -1
        best_fitness_in_tournament = -float('inf')

        for idx in tournament_indices:
            if fitnesses[idx] > best_fitness_in_tournament:
                best_fitness_in_tournament = fitnesses[idx]
                best_idx_in_tournament = idx

        return population[best_idx_in_tournament]

    def _order_crossover(self, parent1, parent2):
        """
        Provede křížení pomocí metody Order Crossover (OX1).
        Zachovává relativní pořadí prvků z druhého rodiče.
        """
        child = [-1] * self.n_cities

        # 1. Vyber náhodný souvislý úsek z parent1
        start, end = sorted(random.sample(range(self.n_cities), 2))
        child[start:end + 1] = parent1[start:end + 1]

        # Množina prvků již přítomných v potomkovi pro rychlou kontrolu
        elements_in_child = set(child[start:end + 1])

        # 2. Doplň zbývající pozice z parent2
        parent2_idx = 0
        child_idx = 0

        while -1 in child:
            # Najdi další volné místo v potomkovi (začínáme od začátku)
            if child[child_idx] == -1:
                # Najdi další prvek v parent2, který ještě není v potomkovi
                current_parent2_element = parent2[parent2_idx]
                while current_parent2_element in elements_in_child:
                    parent2_idx = (parent2_idx + 1) % self.n_cities  # Posun v parent2 (cyklicky)
                    current_parent2_element = parent2[parent2_idx]

                # Vlož prvek a posuň index v parent2
                child[child_idx] = current_parent2_element
                parent2_idx = (parent2_idx + 1) % self.n_cities

            # Posuň index v potomkovi (cyklicky)
            child_idx = (child_idx + 1) % self.n_cities

        return child

    def _swap_mutation(self, route):
        """
        Provede mutaci prohozením dvou náhodně vybraných měst v cestě.
        Mutace se provede s pravděpodobností 'mutation_rate'.
        """
        mutated_route = route[:]  # Vytvoříme kopii
        if random.random() < self.mutation_rate:
            # Vybereme dva různé indexy ke prohození
            idx1, idx2 = random.sample(range(self.n_cities), 2)
            mutated_route[idx1], mutated_route[idx2] = mutated_route[idx2], mutated_route[idx1]
        return mutated_route

    def solve(self):
        """
        Spustí genetický algoritmus a vrátí nejlepší nalezenou cestu a její délku.
        """
        population = self._initialize_population()
        best_overall_route = None
        best_overall_distance = float('inf')

        print(f"Spouštění GA pro TSP: {self.n_generations} generací, velikost populace {self.population_size}...")

        for generation in range(self.n_generations):
            # Vypočítáme fitness pro všechny jedince v populaci
            fitnesses = [self._calculate_fitness(route) for route in population]

            # Najdeme nejlepšího jedince v aktuální generaci
            current_best_idx = np.argmax(fitnesses)
            current_best_route = population[current_best_idx]
            current_best_distance = self._calculate_total_distance(current_best_route)

            # Aktualizujeme celkově nejlepší řešení, pokud je aktuální lepší
            if current_best_distance < best_overall_distance:
                best_overall_distance = current_best_distance
                best_overall_route = current_best_route
                # Vypíšeme informaci o novém nejlepším řešení
                print(f"Generace {generation + 1}: Nová nejlepší vzdálenost = {best_overall_distance:.2f}")
            # Pravidelný výpis progresu, např. každých 20 generací nebo pokud není zlepšení
            elif (generation + 1) % 20 == 0:
                print(f"Generace {generation + 1}: Aktuální nejlepší vzdálenost = {best_overall_distance:.2f}")

            # Vytvoření nové generace
            next_population = []

            # 1. Elitismus: Přeneseme nejlepší jedince přímo
            # Seřadíme indexy jedinců podle fitness sestupně
            sorted_indices = np.argsort(fitnesses)[::-1]
            for i in range(self.elite_size):
                elite_idx = sorted_indices[i]
                next_population.append(population[elite_idx])

            # 2. Doplníme zbytek populace pomocí selekce, křížení a mutace
            while len(next_population) < self.population_size:
                # Selekce rodičů
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                # Můžeme přidat kontrolu, aby rodiče nebyli stejní (není nutné, ale může mírně pomoci diverzitě)
                # while parent1 is parent2:
                #     parent2 = self._tournament_selection(population, fitnesses)

                # Křížení
                child = self._order_crossover(parent1, parent2)

                # Mutace
                mutated_child = self._swap_mutation(child)

                # Přidání nového jedince do další generace
                next_population.append(mutated_child)

            # Nahradíme starou populaci novou
            population = next_population

        print(f"GA dokončen. Nejlepší nalezená vzdálenost: {best_overall_distance:.2f}")
        # Vrátíme nejlepší nalezenou cestu a její vzdálenost
        return best_overall_route, best_overall_distance