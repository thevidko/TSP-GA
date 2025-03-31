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
        image_display = ui.image().style("max-width: 100%;")

    # Prostor pro vykreslení graf

# Funkce pro řešení problému
def solve_tsp():
    distance_matrix = DistanceMatrix(number_of_cities=n_cities.value, min_distance=min_dist.value, max_distance=max_dist.value)
    distance_matrix.print_matrix()
    image_display.set_source("test.jpg")
    print("Solving TSP...")

ui.run(title="TSP Vizualizace")