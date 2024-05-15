"""
 Buliga Theodor Ioan
 Preslav Shterev
 UPM ETSISI - Bioinspired Algorithms for Optimization 2023-2024
"""
import json

import numpy as np
import random
import matplotlib.pyplot as plt
from aco import ACO_UrbanGardening, UrbanGardeningProblem  # Assuming CityLayout is in aco.py


def run_experiment(num_ants, num_iterations, evaporation_rate, alphas, betas, elevation_lists, width, height):
    results = []

    for alpha in alphas:
        for beta in betas:
            for elevation_list in elevation_lists:
                grid_size = len(elevation_list)
                problem = UrbanGardeningProblem(elevation_list, width,
                                                height)  # Initialize UrbanGardeningProblem with elevations
                aco = ACO_UrbanGardening(grid_size, num_ants, num_iterations, problem, evaporation_rate, alpha, beta,
                                         elevation_list, width, height)
                best_solution, best_fitness, _, _, _, _, _, _ = aco.run()
                results.append({
                    'alpha': alpha,
                    'beta': beta,
                    'elevation_list': elevation_list,
                    'best_fitness': best_fitness,
                    'best_solution': best_solution
                })
                print(f"Completed: alpha={alpha}, beta={beta}, best_fitness={best_fitness}")

    return results


def plot_results(results):
    alphas = sorted(set(result['alpha'] for result in results))
    betas = sorted(set(result['beta'] for result in results))

    fig, axes = plt.subplots(len(alphas), len(betas), figsize=(20, 20))
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            ax = axes[i, j]
            fitnesses = [
                result['best_fitness'] for result in results
                if result['alpha'] == alpha and result['beta'] == beta
            ]
            ax.plot(fitnesses, marker='o')
            ax.set_title(f'alpha={alpha}, beta={beta}')
            ax.set_xlabel('Experiment')
            ax.set_ylabel('Best Fitness')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Parameters
    num_ants = 10
    num_iterations = 15
    evaporation_rate = 0.1
    alphas = [0.5, 1, 1.5]
    betas = [0.5, 1, 1.5]

    # Read elevation lists from the input.json file
    with open('input.json', 'r') as f:
        data = json.load(f)
        elevation_lists = data['elevation_lists']  # Make sure this matches your JSON structure

    # Assuming all elevation lists have the same width and height
    width = height = int(np.sqrt(len(elevation_lists[0])))

    results = run_experiment(num_ants, num_iterations, evaporation_rate, alphas, betas, elevation_lists, width, height)

    # Save the results to a JSON file
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)

    plot_results(results)
