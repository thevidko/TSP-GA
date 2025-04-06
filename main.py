from nicegui import ui
import numpy as np
import random
import matplotlib.pyplot as plt
from io import BytesIO

from graph_generator import DistanceMatrix
from solver import TSPGeneticSolver

# UI komponent
# UI rozvržení
with ui.row().style("width: 100%; align-items: flex-start;"):
    with ui.column().style("width: 300px; padding: 20px;"):
        ui.label("GA pro řešení TSP").style("font-size: 24px; font-weight: bold;")
        generate_check = ui.checkbox("Generovat")
        n_cities = ui.number("Počet měst", value=10, min=2, max=100).style("width: 100%").bind_visibility_from(generate_check, 'value')
        min_dist = ui.number("Min. vzdálenost", value=50, min=1, max=2000).style("width: 100%").bind_visibility_from(generate_check, 'value')
        max_dist = ui.number("Max. vzdálenost", value=2000, min=1, max=5000).style("width: 100%").bind_visibility_from(generate_check, 'value')
        file_input = ui.upload(label="Vložit soubor", on_upload=lambda e: ui.notify("Soubor nahrán!")).style("width: 90%")
        solve_button = ui.button("Vyřeš", on_click=lambda: solve_tsp())
        n_generations = ui.number("Počet generací", value= 200, min=1,max=2000).style("width: 100%")
        populationlen = ui.number("Velikost generace", value=100, min=1, max=2000).style("width: 100%")
        mutation_rate = ui.number("Míra mutace", value=0.1, min=0.05, max=0.5,step=0.05).style("width: 100%")
    with ui.column().style("flex-grow: 1; padding-left: 20px;"):
        #image_display = ui.image().style("max-width: 100%;")
        result_label = ui.label("Výsledky se zobrazí zde.")  # Místo pro textový výstup
        route_label = ui.label()
        distance_label = ui.label()

    # Prostor pro vykreslení graf

# Funkce pro řešení problému
def solve_tsp():
    try:
        # 1. Získání matice vzdáleností
        distance_matrix = None  # Inicializace
        if generate_check.value:
            # <<< --- ZDE PŘETYPOVAT HODNOTY NA INT --- >>>
            num_cities = int(n_cities.value)
            min_d = int(min_dist.value)
            max_d = int(max_dist.value)

            if num_cities < 2:
                ui.notify("Počet měst musí být alespoň 2.", type='warning')
                return

            distance_matrix = DistanceMatrix(
                number_of_cities=num_cities,
                min_distance=min_d,
                max_distance=max_d
            )
            print("Vygenerovaná matice vzdáleností:")
            distance_matrix.print_matrix()
        else:
            # Zde by byla logika pro načtení distance_matrix z file_input
            ui.notify("Načítání ze souboru ještě není implementováno.", type='warning')
            print("Načítání ze souboru není implementováno.")
            result_label.set_text("Chyba: Načítání ze souboru není implementováno.")
            return  # Ukončit funkci, pokud nemáme matici

        # Kontrola, zda máme platnou matici
        if distance_matrix is None:
            ui.notify("Nebyla vytvořena nebo načtena matice vzdáleností.", type='error')
            result_label.set_text("Chyba: Chybí matice vzdáleností.")
            return

        # 2. Vytvoření instance solveru
        # <<< --- ZDE PŘETYPOVAT HODNOTY NA INT --- >>>
        pop_size = int(populationlen.value)
        num_gens = int(n_generations.value)

        # Kontrola smysluplnosti parametrů
        if pop_size <= 0 or num_gens <= 0:
            ui.notify("Velikost populace a počet generací musí být kladné.", type='warning')
            return
        # Můžete přidat i kontrolu pro mutation_rate, i když je to float

        solver = TSPGeneticSolver(
            distance_matrix=distance_matrix,
            population_size=pop_size,
            mutation_rate=mutation_rate.value,  # mutation_rate může být float
            n_generations=num_gens
            # elite_size a tournament_size mají výchozí hodnoty v solveru
        )

        # 3. Spuštění řešení
        print("Spouštění genetického algoritmu...")
        result_label.set_text("Probíhá výpočet...")
        # UI může během výpočtu zamrznout, pro delší výpočty zvážit spuštění v threadu
        best_route, best_distance = solver.solve()

        # 4. Zobrazení výsledků
        print("Řešení dokončeno.")

        if best_route:
            print(f"Nejlepší cesta: {best_route}")
            print(f"Nejlepší vzdálenost: {best_distance}")
            result_label.set_text("Výpočet dokončen.")
            route_label.set_text(f"Nejlepší nalezená cesta: {best_route}")
            distance_label.set_text(f"Celková vzdálenost: {best_distance:.2f}")
            ui.notify(f"Nalezena cesta s délkou {best_distance:.2f}!", type='positive')
            # TODO: Vizualizace cesty
        else:
            print("Nepodařilo se nalézt řešení.")
            result_label.set_text("Nepodařilo se nalézt řešení.")
            ui.notify("Algoritmus nedokončil hledání řešení.", type='warning')


    except ValueError as e:
        # Chyby při přetypování nebo jiné ValueError
        print(f"Chyba hodnoty: {e}")
        ui.notify(f"Chyba vstupní hodnoty: {e}", type='negative')
        result_label.set_text(f"Chyba vstupu: {e}")
    except Exception as e:
        # Zachycení všech ostatních neočekávaných chyb
        print(f"Nastala neočekávaná chyba: {e}")
        # Zobrazit typ chyby pro lepší diagnostiku
        error_type = type(e).__name__
        print(f"Typ chyby: {error_type}")
        ui.notify(f"Neočekávaná chyba ({error_type}): {e}", type='negative')
        result_label.set_text(f"Neočekávaná chyba: {e}")

ui.run(title="TSP Vizualizace")