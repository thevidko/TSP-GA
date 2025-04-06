import base64
from nicegui import ui
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from graph_generator import DistanceMatrix
from solver import TSPGeneticSolver

# UI
with ui.row().style("width: 100%; align-items: flex-start;"):
    with ui.column().style("width: 300px; padding: 20px;"):
        ui.label("GA pro řešení TSP").style("font-size: 24px; font-weight: bold;")
        generate_check = ui.checkbox("Generovat")
        n_cities = ui.number("Počet měst", value=10, min=2, max=1000).style("width: 100%").bind_visibility_from(generate_check, 'value')
        min_dist = ui.number("Min. vzdálenost", value=50, min=1, max=2000).style("width: 100%").bind_visibility_from(generate_check, 'value')
        max_dist = ui.number("Max. vzdálenost", value=2000, min=1, max=10000).style("width: 100%").bind_visibility_from(generate_check, 'value')
        file_input = ui.upload(label="Vložit soubor", on_upload=lambda e: ui.notify("Soubor nahrán!")).style("width: 90%")
        solve_button = ui.button("Vyřeš", on_click=lambda: solve_tsp())
        n_generations = ui.number("Počet generací", value= 200, min=1,max=2000).style("width: 100%")
        populationlen = ui.number("Velikost generace", value=100, min=1, max=2000).style("width: 100%")
        mutation_rate = ui.number("Míra mutace", value=0.1, min=0.05, max=0.5,step=0.05).style("width: 100%")
    with ui.column().style("flex-grow: 1; padding-left: 20px;"):
        result_label = ui.label("Výsledky se zobrazí zde.")  # Místo pro textový výstup
        route_label = ui.label()
        distance_label = ui.label()
        image_display = ui.image().style("max-width: 100%; border: 1px solid lightgray;")

def plot_tsp_route(coordinates, route, title="TSP - Nejlepší cesta"):
    """
    Vykreslí města, všechny možné spoje (nevýrazně) a nejlepší nalezenou cestu.

    Args:
        coordinates (np.ndarray): Pole (n_cities, 2) se souřadnicemi (x, y).
        route (list): Seznam indexů měst v pořadí nejlepší cesty.
        title (str): Název grafu.

    Returns:
        str: Data URI (base64) PNG obrázku, nebo None pokud nastane chyba.
    """
    if coordinates is None or coordinates.shape[0] < 2 :
        print("Chyba: Chybí nebo jsou nedostatečné souřadnice pro vykreslení.")
        return None
    if route is None or len(route) == 0:
         print("Varování: Chybí data o cestě pro vykreslení (vykreslí se jen města a všechny spoje).")

    num_cities = coordinates.shape[0]
    if route and num_cities != len(route):
         print(f"Chyba: Nesoulad počtu měst v cestě ({len(route)}) a souřadnicích ({num_cities}).")
         route = None #

    try:
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
             print("Styl 'seaborn-v0_8-darkgrid' není dostupný, používá se výchozí.")

        fig, ax = plt.subplots(figsize=(8, 8))

        x_coords = coordinates[:, 0]
        y_coords = coordinates[:, 1]

        # Vykreslení všech možných spojů
        added_legend_all = False # Pomocná proměnná pro přidání legendy jen jednou
        for i in range(num_cities):
            for j in range(i + 1, num_cities): # Iterujeme přes všechny unikátní páry měst
                label_all = 'Všechny možné spoje' if not added_legend_all else None
                ax.plot([x_coords[i], x_coords[j]], # X souřadnice [start, end]
                        [y_coords[i], y_coords[j]], # Y souřadnice [start, end]
                        color='gray',
                        linestyle=':',
                        linewidth=0.7,
                        alpha=1,
                        zorder=1,
                        label=label_all)
                if not added_legend_all:
                    added_legend_all = True

        # 1. Vykreslení měst (jako body)
        ax.scatter(x_coords, y_coords, c='dodgerblue', s=150, zorder=5, label='Města')

        # 2. Přidání čísel k městům
        for i in range(num_cities):
            ax.text(x_coords[i], y_coords[i] + 0.01 * (coordinates[:,1].max() - coordinates[:,1].min()),
                    str(i), fontsize=10, color='black', zorder=6, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3, ec='none')) # Žluté pozadí pro čísla

        # 3. Vykreslení nejlepší cesty (pokud existuje)
        if route:
            route_x = [coordinates[city_index][0] for city_index in route]
            route_y = [coordinates[city_index][1] for city_index in route]
            # Přidáme startovní město na konec pro uzavření smyčky
            route_x.append(coordinates[route[0]][0])
            route_y.append(coordinates[route[0]][1])

            ax.plot(route_x, route_y, 'r-', linewidth=2.5, label='Nejlepší cesta', zorder=4) # Červená čára

        # Nastavení grafu (titulek, popisky os, legenda, mřížka, poměr stran)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("X souřadnice", fontsize=12)
        ax.set_ylabel("Y souřadnice", fontsize=12)
        ax.legend() # Legenda se zobrazí pro prvky, které mají 'label'
        ax.grid(True, linestyle='--', alpha=0.6, zorder=0) # Mřížka úplně vespod

        # Nastavení rozsahu os pro lepší zobrazení
        padding_x = (x_coords.max() - x_coords.min()) * 0.05 # 5% okraj
        padding_y = (y_coords.max() - y_coords.min()) * 0.05
        ax.set_xlim(x_coords.min() - padding_x, x_coords.max() + padding_x)
        ax.set_ylim(y_coords.min() - padding_y, y_coords.max() + padding_y)
        ax.set_aspect('equal', adjustable='box') # Zajistí stejné měřítko os

        # Uložení obrázku do paměti (BytesIO)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        # Převod na base64 Data URI
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        data_uri = f"data:image/png;base64,{img_str}"

        plt.close(fig) # Zavřeme plot
        print("Obrázek grafu (včetně všech spojů) úspěšně vygenerován do paměti.")
        return data_uri

    except Exception as e:
        print(f"Chyba při generování grafu: {e}")
        # zavření plot i v případě chyby
        if 'fig' in locals():
            plt.close(fig)
        return None


# Funkce pro řešení problému
def solve_tsp():
    try:
        # 1. Získání matice vzdáleností
        distance_matrix = None  # Inicializace
        if generate_check.value:
            num_cities = int(n_cities.value)
            max_d = int(max_dist.value)

            if num_cities < 2:
                ui.notify("Počet měst musí být alespoň 2.", type='warning')
                return

            distance_matrix = DistanceMatrix(
                number_of_cities=num_cities,
                coord_range_max=max_d
            )
            print("Vygenerovaná matice vzdáleností:")
            distance_matrix.print_matrix()
        else:
            # TODO: logika pro načtení distance_matrix z file_input
            ui.notify("Načítání ze souboru ještě není implementováno.", type='warning')
            print("Načítání ze souboru není implementováno.")
            result_label.set_text("Chyba: Načítání ze souboru není implementováno.")
            return

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
        # TODO: spustit v jiném threadu
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
            # VYKRESLENÍ A ZOBRAZENÍ GRAFU
            coords = distance_matrix.get_coordinates()
            image_data_uri = plot_tsp_route(coords, best_route)
            if image_data_uri:
                image_display.set_source(image_data_uri)
            else:
                image_display.set_source('')  # Vymaže obrázek, pokud se nepovedl
                ui.notify("Nepodařilo se vygenerovat obrázek grafu.", type='warning')
        else:
            print("Nepodařilo se nalézt řešení.")
            result_label.set_text("Nepodařilo se nalézt řešení.")
            image_display.set_source('')
            ui.notify("Algoritmus nedokončil hledání řešení.", type='warning')


    except ValueError as e:
        # Chyby při přetypování nebo jiné ValueError
        print(f"Chyba hodnoty: {e}")
        ui.notify(f"Chyba vstupní hodnoty: {e}", type='negative')
        result_label.set_text(f"Chyba vstupu: {e}")
        image_display.set_source('')
    except Exception as e:
        # Zachycení všech ostatních neočekávaných chyb
        print(f"Nastala neočekávaná chyba: {e}")
        # Zobrazit typ chyby pro lepší diagnostiku
        error_type = type(e).__name__
        print(f"Typ chyby: {error_type}")
        ui.notify(f"Neočekávaná chyba ({error_type}): {e}", type='negative')
        result_label.set_text(f"Neočekávaná chyba: {e}")
        image_display.set_source('')

ui.run(title="TSP Vizualizace")