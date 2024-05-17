"""
 Buliga Theodor Ioan
 UPM ETSISI - Bioinspired Algorithms for Optimization 2023-2024
"""

from inspyred import ec, benchmarks
import random
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from fitness import CityLayout
import time

"""
 ACO implementation
"""
# Subclass to be used by the ACO class
class UrbanGardeningProblem:
    def __init__(self, elevations, width, height):
        self.elevations = elevations
        self.width = width
        self.height = height

    def evaluate(self, solution):
        # Evaluates the urban layout solution using the fitness function from CityLayout
        city_layout = CityLayout(self.elevations, self.width, self.height)
        return city_layout.calculate_fitness(solution, self.elevations)

# The class for testing
class ACO_UrbanGardening:
    def __init__(self, grid_size, num_ants, num_iterations, problem, evaporation_rate, alpha, beta, elevations_string, width, height):
        self.grid_size = grid_size # Grid dimension
        self.num_ants = num_ants # The number of ants
        self.num_iterations = num_iterations # Total iterations
        self.problem = problem
        self.pheromones = np.ones((grid_size, 4))  # Pheromone levels for each tile type at each grid position
        self.evaporation_rate = evaporation_rate # The evaporation rate
        self.alpha = alpha  # Pheromone influence
        self.beta = beta    # Heuristic influence
        self.tile_types = ['R', 'C', 'S', 'G']  # Residential, Commercial, Streets, Green spaces
        self.elevations_string = elevations_string
        self.width = width # Width
        self.height = height # Height

        # History tracking
        self.history = []
        self.solution_history_fitness = []

        self.commercial_weight_history = []
        self.green_weight_history = []
        self.res_weight_history = []
        self.street_adjacency_history = []
        self.nearby_green_weight_history = []
        self.street_connectivity_history = []
        self.elev_weight_overall_history = []
        self.street_weight_overall_history = []
        self.res_cluster_weight_history = []

        self.best_fitness_historic = []
        self.diversity_historic = []
        self.city_layout = CityLayout(self.elevations_string, self.width, self.height)
        self.trails_history = []


    # Display the solution as a matrix
    def visualize_solution(self, solution, best_fitness):
        color_map = {'C': 'blue', 'R': 'brown', 'S': 'grey', 'G': 'green'}
        color_array = [color_map[tile] for tile in solution]

        fig, ax = plt.subplots(figsize=(10, 12))
        for i in range(self.width):
            for j in range(self.height):
                index = i * self.width + j
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color_array[index]))

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.invert_yaxis()
        ax.axis('off')
        # Display configuration details below the matrix
        config_text = (
            f"Alpha: {self.alpha}\n"
            f"Beta: {self.beta}\n"
            f"Evaporation Rate: {self.evaporation_rate}\n"
            f"Number of Ants: {self.num_ants}\n"
            f"Best Fitness: {best_fitness}\n"
            f"Grid Dimension: {self.width}x{self.height}"
        )
        plt.figtext(0.5, 0.02, config_text, ha="center", fontsize=12,
                    bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
        plt.show()

    # Apply the algorithm
    def run(self):
        start_time = time.time()  # Capture the start time
        best_solution = None # No solution yet
        best_fitness = -np.inf # No fitness yet
        self.trails_history = []  # List to store fitness of each ant per iteration

        # Start of the algorithm
        for iteration in range(self.num_iterations):
            solutions = [self.generate_solution() for _ in range(self.num_ants)]
            fitnesses = [self.problem.evaluate(solution) for solution in solutions]

            self.trails_history.append([(solution, fitness) for solution, fitness in zip(solutions, fitnesses)])

            self.record_history(solutions, fitnesses)  # Record the evolution history
            # Update pheromones globally
            self.update_global_pheromones(solutions, fitnesses)

            # Check for new best solution
            current_best_index = np.argmax(fitnesses)
            if fitnesses[current_best_index] > best_fitness:
                best_fitness = fitnesses[current_best_index]
                best_solution = solutions[current_best_index]

        end_time = time.time()  # Capture the end time
        execution_time = end_time - start_time  # Calculate the execution time

        self.visualize_solution(best_solution, best_fitness)

        self.visualize_fitness_history()

        self.visualize_diversity()

        self.visualize_parameter_evolution(execution_time)

        return (best_solution, best_fitness, self.commercial_weight_history,
                self.green_weight_history, self.res_weight_history, self.street_adjacency_history,
                self.street_connectivity_history,
                self.elev_weight_overall_history, execution_time)


    # Keep track of the history for comparison
    def record_history(self, solutions, fitnesses):
        self.history.append(solutions)
        best_index = np.argmax(fitnesses)
        best_solution = solutions[best_index]

        self.solution_history_fitness.append(self.city_layout.calculate_fitness(best_solution, self.problem.elevations))
        self.commercial_weight_history.append(self.city_layout.commercial_weight(best_solution))
        self.green_weight_history.append(self.city_layout.green_weight(best_solution))
        self.res_weight_history.append(self.city_layout.res_weight(best_solution))
        self.street_adjacency_history.append(self.city_layout.street_adjacency_weight(best_solution))
        self.nearby_green_weight_history.append(self.city_layout.nearby_green_weight(best_solution))
        self.street_connectivity_history.append(self.city_layout.street_connectivity_weight(best_solution))
        self.elev_weight_overall_history.append(self.city_layout.elev_weight_normal(best_solution, self.problem.elevations))
        self.street_weight_overall_history.append(self.city_layout.street_weight(best_solution))
        self.res_cluster_weight_history.append(self.city_layout.res_clusters_weight(best_solution))

        self.best_fitness_historic.append(fitnesses[best_index])
        self.diversity_historic.append(self.calculate_diversity(solutions))

    # Calculate how diverse it is
    def calculate_diversity(self, solutions):
        value_map = {'R': 0, 'S': 1, 'C': 2, 'G': 3}
        numerical_solutions = [[value_map[tile] for tile in solution] for solution in solutions]
        return np.std(numerical_solutions, axis=0).mean()

    # Display the grafic
    def visualize_fitness_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.best_fitness_historic, label='Best Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Best Fitness Over Generations')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Helper function #modularization
    def generate_solution(self):
        solution = []
        for i in range(self.grid_size):
            next_tile = self.choose_next_tile(i, solution)
            solution.append(next_tile)
        return solution

    # Next tile choice
    def choose_next_tile(self, position, current_solution):
        heuristic_values = self.calculate_heuristic(position, current_solution)

        # Ensure pheromones array has the same shape as heuristic values
        pheromones = self.pheromones[position]
        if len(pheromones) != len(heuristic_values):
            raise ValueError("Pheromone and heuristic arrays must have the same length")

        # Calculate the current counts of each tile type in the solution
        count_R = current_solution.count('R')
        count_C = current_solution.count('C')
        count_G = current_solution.count('G')
        count_S = current_solution.count('S')

        # Desired proportions
        desired_R_min = self.grid_size * 0.20
        desired_R_max = self.grid_size * 0.30
        desired_C_min = self.grid_size * 0.10
        desired_C_max = self.grid_size * 0.20
        desired_G_min = self.grid_size * 0.15
        desired_G_max = self.grid_size * 0.25
        desired_S_min = self.grid_size * 0.20
        desired_S_max = self.grid_size * 0.30

        # Adjust heuristic values based on current proportions
        def adjust_heuristic(count, desired_min, desired_max, heuristic, tile_type):
            if count < desired_min:
                return heuristic * (1 + (desired_min - count) / self.grid_size)
            elif count > desired_max:
                return 0 if tile_type else heuristic * (1 - (count - desired_max) / self.grid_size)
            return heuristic

        # Adjust heuristic values based on current proportions
        heuristic_values[0] = adjust_heuristic(count_R, desired_R_min, desired_R_max, heuristic_values[0], 'R')
        heuristic_values[1] = adjust_heuristic(count_C, desired_C_min, desired_C_max, heuristic_values[1], 'C')
        heuristic_values[2] = adjust_heuristic(count_S, desired_S_min, desired_S_max, heuristic_values[2], 'S')
        heuristic_values[3] = adjust_heuristic(count_G, desired_G_min, desired_G_max, heuristic_values[3], 'G')

        # Calculate scores using both pheromone and heuristic information
        scores = (np.array(pheromones) ** self.alpha) * (np.array(heuristic_values) ** self.beta)
        if np.any(scores < 0) or np.sum(scores) <= 0:
            #print(
                #f"Debug: Negative or zero scores encountered. Scores: {scores}, alpha: {self.alpha}, beta: {self.beta}")
            scores[scores < 0] = 0  # Set negative scores to zero
        # Normalize scores to get probabilities
        if scores.sum() > 0:
            probabilities = scores / scores.sum()
        else:
            probabilities = np.ones_like(scores) / len(scores)


        # Select next tile based on the highest score
        next_tile = np.random.choice(self.tile_types, p=probabilities)
        return next_tile

    """
     !IMPORTANT!
    
     The heuristic function 
     used by the algorithm
    """
    def calculate_heuristic(self, position, solution):
        if not solution:
            return np.array([1, 1, 1, 1])  # Return neutral heuristic values

        grid_matrix = [solution[i:i + self.width] for i in range(0, len(solution), self.width)]
        i, j = divmod(position, self.width)

        # Neighborhood analysis with boundary checks
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E
        neighbors = [
            (i + di, j + dj)
            for di, dj in directions
            if 0 <= i + di < self.height and 0 <= j + dj < self.width
        ]

        # Filter neighbors that are within the bounds of the solution matrix
        valid_neighbors = [(ni, nj) for ni, nj in neighbors if 0 <= ni < self.height and 0 <= nj < self.width]

        # Access neighboring tiles only if they exist in the solution matrix
        neighbor_tiles = []
        for ni, nj in valid_neighbors:
            if ni < len(grid_matrix) and nj < len(grid_matrix[ni]):
                neighbor_tiles.append(grid_matrix[ni][nj])
            else:
                neighbor_tiles.append(None)  # Placeholder for out-of-bounds indices

        # Heuristic calculation for tile proportions
        H_residential = sum(1 for t in neighbor_tiles if t == 'R')  # Residential tiles
        H_commercial = sum(1 for t in neighbor_tiles if t == 'C')  # Commercial tiles
        H_green = sum(1 for t in neighbor_tiles if t == 'G')  # Green tiles
        H_street = sum(1 for t in neighbor_tiles if t == 'S')  # Street tiles

        # Additional heuristics
        H_street_adj = 1 if 'S' in neighbor_tiles else 0  # Adjacent street tiles
        H_res_clusters = 1 if 2 <= H_residential <= 10 else 0.5  # Residential clusters size
        H_nearby_green = 1 if any(t == 'G' for t in neighbor_tiles) else 0
        H_street_connectivity = 1 if 'S' in neighbor_tiles else 0

        # Increase weight for street connectivity
        if H_street_connectivity > 0:
            H_street_connectivity = 2  # Boost the heuristic value if there are adjacent streets

        highest_point = max(self.elevations_string)
        lowest_point = min(self.elevations_string)
        elevation_threshold_R = (highest_point - lowest_point) / 3
        elevation_threshold_C = (highest_point - lowest_point) / 5
        elevation_threshold_G = (highest_point - lowest_point) / 3

        elevation = self.elevations_string[position]

        H_elev_weight_R = 1 if elevation <= elevation_threshold_R else 0
        H_elev_weight_C = 1 if elevation <= elevation_threshold_C else 0
        H_elev_weight_G = 1 if elevation >= elevation_threshold_G else 0

        # Aggregate heuristics to match tile types
        H_R = (H_residential + H_res_clusters + H_nearby_green + H_street_adj + H_elev_weight_R) / 5
        H_C = (H_commercial + H_elev_weight_C) / 2
        H_S = (H_street + H_street_connectivity) / 2
        H_G = (H_green + H_elev_weight_G) / 2

        # Normalize
        heuristics = np.array([H_R, H_C, H_S, H_G])
        max_value = np.max(heuristics)
        if max_value > 0:
            heuristics /= max_value  # Avoid division by zero

        return heuristics

    def update_global_pheromones(self, solutions, fitnesses):
        # Evaporate pheromones
        self.pheromones *= (1 - self.evaporation_rate)

        # Add pheromone based on quality of solutions
        for solution, fitness in zip(solutions, fitnesses):
            for idx, tile in enumerate(solution):
                tile_index = self.tile_types.index(tile)
                self.pheromones[idx, tile_index] += fitness

    def visualize_fitness_history(self):
        sns.set_style('darkgrid')
        fitness = np.array([[ant[1] for ant in trails] for trails in self.trails_history])
        best_fitness = np.array(self.solution_history_fitness)

        fig, axs = plt.subplots(figsize=(10, 5))
        axs.set_title('Fitness Evolution')
        axs.set_xlabel('Iterations')
        axs.set_ylabel('Fitness')

        median = np.median(fitness, axis=1)
        min_val = np.min(fitness, axis=1)
        max_val = np.max(fitness, axis=1)

        axs.plot(best_fitness, label='Best Fitness', color='blue')
        axs.plot(median, label='Median Fitness', color='orange')
        axs.fill_between(np.arange(len(median)), min_val, max_val, alpha=0.3, color='orange')

        plt.legend()
        plt.show()

    def visualize_diversity(self):
        sns.set_style('darkgrid')
        fig, axs = plt.subplots(figsize=(10, 5))
        axs.set_title('Diversity Evolution')
        axs.set_xlabel('Iterations')
        axs.set_ylabel('Diversity')

        axs.plot(self.diversity_historic, color='orange')

        plt.show()

    def visualize_parameter_evolution(self, execution_time):
        plt.figure(figsize=(20, 20))
        plt.plot(self.commercial_weight_history, label='Commercial Weight')
        plt.plot(self.green_weight_history, label='Green Weight')
        plt.plot(self.res_weight_history, label='Residential Weight')
        plt.plot(self.street_adjacency_history, label='Street Adjacency')
        plt.plot(self.nearby_green_weight_history, label='Nearby Green')
        plt.plot(self.street_connectivity_history, label='Street Connectivity')
        plt.plot(self.elev_weight_overall_history, label='Elevations Overall')
        plt.plot(self.res_cluster_weight_history, label='Cluster Weight')
        plt.plot(self.street_weight_overall_history, label='Street Weight')

        plt.xlabel('Generation')
        plt.ylabel('Weight Value')
        plt.title('Parameter Evolution Over Generations')

        # Annotations for parameters
        param_text = (f"Alpha: {self.alpha}\nBeta: {self.beta}\n"
                      f"Num Ants: {self.num_ants}\n"
                      f"Evaporation Rate: {self.evaporation_rate}\n"
                      f"Execution Time: {execution_time:.2f} seconds")
        plt.annotate(param_text, xy=(0.01, 0.99), xycoords='axes fraction',
                     textcoords='axes fraction', va='top', ha='left', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.5))

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
        plt.show()