import numpy as np
import random
import matplotlib.pyplot as plt
from common import ACO_UrbanGardening, UrbanGardeningProblem  # Assuming CityLayout is in common.py

def run_experiment(num_ants, num_iterations, evaporation_rate, alphas, betas, elevation_lists):
    results = []

    for alpha in alphas:
        for beta in betas:
            for elevation_list in elevation_lists:
                grid_size = len(elevation_list)
                width = int(np.sqrt(grid_size))
                height = grid_size // width

                problem = UrbanGardeningProblem(elevation_list, width, height)  # Initialize UrbanGardeningProblem with elevations
                aco = ACO_UrbanGardening(grid_size, num_ants, num_iterations, problem, evaporation_rate, alpha, beta, elevation_list, width, height)
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
    num_ants = 10
    num_iterations = 15
    evaporation_rate = 0.1

    alphas = [0.5, 1, 1.5]
    betas = [0.5, 1, 1.5]

    # Provided elevation lists
    elevation_lists = [
        np.array([
            [100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
            [150, 200, 250, 300, 350, 400, 450, 500, 550, 600],
            [200, 250, 300, 350, 400, 450, 500, 550, 600, 650],
            [250, 300, 350, 400, 450, 500, 550, 600, 650, 700],
            [300, 350, 400, 450, 500, 550, 600, 650, 700, 750],
            [350, 400, 450, 500, 550, 600, 650, 700, 750, 800],
            [400, 450, 500, 550, 600, 650, 700, 750, 800, 850],
            [450, 500, 550, 600, 650, 700, 750, 800, 850, 900],
            [500, 550, 600, 650, 700, 750, 800, 850, 900, 950],
            [550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
        ]).flatten().tolist(),
        np.array([
            [500, 550, 600, 650, 700, 750, 800, 850, 900, 950],
            [550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
            [600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050],
            [650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100],
            [700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150],
            [750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200],
            [800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250],
            [850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300],
            [900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350],
            [950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400]
        ]).flatten().tolist(),
        [300, 450, 600, 750, 600, 450, 450, 600, 750, 900, 750, 600,
         600, 750, 900, 1050, 900, 750, 750, 900, 1050, 1200, 1050, 900,
         600, 750, 900, 1050, 900, 750, 450, 600, 750, 900, 750, 600],
        [502, 628, 735, 799, 926, 1052, 1203, 1351, 371, 481, 589, 662, 798, 912, 1067, 1214,
         244, 360, 459, 561, 701, 822, 958, 1123, 122, 238, 344, 434, 579, 701, 839, 987,
         5, 120, 220, 326, 456, 580, 725, 891, 0, 90, 184, 278, 385, 502, 650, 811,
         0, 10, 101, 188, 299, 407, 545, 702, 0, 0, 51, 130, 243, 355, 492, 640],
        [445, 242, 273, 280, 224, 239, 484, 465, 424, 382, 430, 363, 227, 230, 471, 364, 433,
         227, 234, 402, 331, 311, 404, 255, 495, 373, 368, 414, 360, 219, 261, 239, 332,
         466, 484, 398, 488, 229, 416, 396, 219, 347, 430, 409, 452, 418, 485, 239, 438],
        [200, 200, 200, 200, 200, 200, 200, 200, 400, 400, 400, 400, 400, 200, 200, 400, 600,
         600, 600, 400, 200, 200, 400, 600, 800, 600, 400, 200, 200, 400, 600, 600, 600, 400,
         200, 200, 400, 400, 400, 400, 400, 200, 200, 200, 200, 200, 200, 200, 200],
        [500, 550, 600, 650, 700, 650, 600, 550, 550, 600, 650, 700, 750, 700, 650, 600,
         600, 650, 700, 750, 800, 750, 700, 650, 650, 700, 750, 800, 850, 800, 750, 700,
         700, 750, 800, 850, 900, 850, 800, 750, 650, 700, 750, 800, 850, 800, 750, 700,
         600, 650, 700, 750, 800, 750, 700, 650, 550, 600, 650, 700, 750, 700, 650, 600]
    ]

    results = run_experiment(num_ants, num_iterations, evaporation_rate, alphas, betas, elevation_lists)
    plot_results(results)
