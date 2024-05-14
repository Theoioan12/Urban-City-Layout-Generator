# Imports for all of the code
import math
from random import Random
from time import time
from inspyred import ec, benchmarks
from matplotlib import pyplot as plt

# Static variables
gridsize = 100
height = 10
width = 10

import random
# Generate 16 random values between 0 and 1000
elevations_string = [random.randint(0, 1000) for _ in range(gridsize)]
#elevations_string = [948,924,1000,903,936,  935,  990,  939,   68   43   40   96   71  968
  #938   43   25   97   16   88  990  901   99   14   67   22    9  909
  #940   84    0   81   82   76  985  946   95   80    1   46   80  993
  #956  915  981  932  996  993  982]

import numpy as np
import matplotlib.pyplot as plt

def generate_elevation_matrix(size, max_altitude):
    elevation_matrix = np.zeros((size, size))

    # Define altitude gradient parameters
    center_altitude = max_altitude / 2
    edge_altitude = max_altitude

    # Compute altitude gradient
    for i in range(size):
        for j in range(size):
            distance_to_center = min(i, j, size - i - 1, size - j - 1)
            elevation_matrix[i, j] = center_altitude + (edge_altitude - center_altitude) * (distance_to_center / (size / 2))

    return elevation_matrix

# Example usage
size = width  # Size of the elevation matrix (square)
max_altitude = 1500  # Maximum altitude in meters
elevation_matrix = generate_elevation_matrix(size, max_altitude)

elevations_string = elevation_matrix.flatten()
print(elevations_string)
#elevations_string = [850, 845, 830, 800, 750, 850, 845, 830, 800, 750, 850, 845, 830, 800, 750, 850, 845, 830, 800, 750, 850, 845, 830, 800, 750]


# city layout to be used for generation
class CityLayout(benchmarks.Benchmark):
  # init method
  def _init_(self, elevations_string):
        benchmarks.Benchmark._init_(self, len(elevations_string))
        #self.init_layout = init_layout
        self.elevations_string = elevations_string
        self.bounder = ec.DiscreteBounder([0, 1])
        self.maximize = True
        self.best_all = None
        self.best_feasible = None
  def generator(self, random, args):
        #print("calls generator")
        """Return a candidate solution for an evolutionary algorithm.""" #TODO: adapt in order of our needs
        chars = ['R', 'C', 'S', 'G']
        # Generate a random array of 36 characters
        layout = [random.choice(chars) for _ in range(gridsize)]
        #print("it will return from generator" + str(layout) )
        return layout

  def evaluator(self, candidates, args):
        #print("Calls evaluator")

        """Return the fitness values for the given candidates."""
        fitness = []
        for candidate in candidates:
            #print("candidate is: " + str(candidate))
            fitness.append(CityLayout.calculate_fitness(candidate, self.elevations_string))
        #print("it will return from evaluator" + str(fitness) )
        #print("best fitness is: " + str(max(fitness)))

        return fitness

  def death_penalty_evaluator(self, candidates, args):
        """Evaluate the fitness of the given candidates with a death penalty."""
        fitness = []
        for c in candidates:
            #if CityLayout.street_adjacency_weight(c) < 1 or CityLayout.commercial_weight(c) < 1 or CityLayout.elev_weight_normal(c, self.elevations_string) < 1 or CityLayout.green_weight(c) < 1 or CityLayout.nearby_green_weight(c) < 1 or CityLayout.res_weight(c) < 1 or CityLayout.street_connectivity_weight(c) < 1:
            if CityLayout.street_connectivity_weight(c) < 0.6:
                fit = 0
            else:
                fit = CityLayout.calculate_fitness(c, self.elevations_string)
            fitness.append(fit)
        return fitness

  def static_penalty_evaluator(self, candidates, args):
        """Evaluate the fitness of the given candidates with a static penalty."""
        Rg = args.get('Rg', 3)
        Rh = args.get('Rh', 10)
        fitness = []
        for c in candidates:
            fit = CityLayout.calculate_fitness(c, self.elevations_string) - Rg*max(0, CityLayout.res_weight(c)) - Rh*max(0, CityLayout.street_connectivity_weight(c))
            fitness.append(fit)
        return fitness

  def dynamic_penalty_evaluator(self, candidates, args):
        """Evaluate the fitness of the given candidates with a dynamic penalty."""
        C = args.get('p_c', 5)
        alpha = args.get('p_alpha', 2)
        beta = args.get('p_beta', 2)
        t = args['_ec'].num_generations
        fitness = []
        for c in candidates:
            viol = CityLayout.street_adjacency_weight(c) * beta + CityLayout.commercial_weight(c) * beta + CityLayout.elev_weight_normal(c, self.elevations_string) * beta + CityLayout.green_weight(c) *beta + CityLayout.nearby_green_weight(c) * beta + CityLayout.res_weight(c) * beta + CityLayout.street_connectivity_weight(c)*beta
            fit = CityLayout.calculate_fitness(c, self.elevations_string) - (C*t)**alpha * viol
            fitness.append(fit)
        return fitness

  def smith_adaptive_penalty_evaluator(self, candidates, args):
        """Evaluate the fitness of the given candidates with a Smith's adaptive penalty."""
        nft0 = args.get('NFT0', 0.01)
        lmbd = args.get('lambda', 1)
        beta = args.get('p_beta', 1)
        t = args['_ec'].num_generations
        nft = nft0 / (1 + lmbd*t)
        viol = []
        fit = []
        for c in candidates:
            v = 7 - CityLayout.street_adjacency_weight(c) + CityLayout.commercial_weight(c) + CityLayout.elev_weight_normal(c, self.elevations_string) + CityLayout.green_weight(c) + CityLayout.nearby_green_weight(c) + CityLayout.res_weight(c) + CityLayout.street_connectivity_weight(c)
            viol.append(v)
            fit.append(CityLayout.calculate_fitness(c, self.elevations_string))
        max_fit = max(fit)
        if self.best_all is None:
            self.best_all = max_fit
        elif max_fit > self.best_all:
            self.best_all = max_fit
        it = [f for f, v in zip(fit, viol) if v == 0]
        if it:
            max_feasible = max(it)
        else:
            max_feasible = 0
        if self.best_feasible is None:
            self.best_feasible = max_feasible
        elif max_feasible > self.best_feasible:
            self.best_feasible = max_feasible
        fitness = []
        for f, v in zip(fit, viol):
            fitness.append(f + (self.best_feasible - self.best_all) * v)
        return fitness

  def find_groups(grid): #searches for groups of R and C
    def dfs(i, j, group):
      if 0 <= i < height and 0 <= j < width and visited[i][j] == False and grid_matrix[i][j] in ['R']:
        visited[i][j] = True
        #print(str(i) + str(j) + ":" + str(grid_matrix[i][j]) + " was marked visited")
        group.append((i, j))
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
          dfs(i + di, j + dj, group)
      elif 0 <= i < height and 0 <= j < width and grid_matrix[i][j] == 'S':
        group.append(True)


    grid_matrix = [grid[i:i+width] for i in range(0, len(grid), width)]
    visited = [[False] * width for _ in range(height)]
    groups = []

    for i in range(height):
      for j in range(width):
        if visited[i][j] == False and grid_matrix[i][j] in ['R']:
          group = []
          dfs(i, j, group)
          if group:
            groups.append(group)

    return groups

  def find_streets_connected(grid): #searches for groups of R and C
    def dfs(i, j, group):
      if 0 <= i < height and 0 <= j < width and visited[i][j] == False and grid_matrix[i][j] in ['S']:
        visited[i][j] = True
        #print(str(i) + str(j) + ":" + str(grid_matrix[i][j]) + " was marked visited")
        group.append((i, j))
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
          dfs(i + di, j + dj, group)


    grid_matrix = [grid[i:i+width] for i in range(0, len(grid), width)]
    visited = [[False] * width for _ in range(height)]
    groups = []

    for i in range(height):
      for j in range(width):
        if visited[i][j] == False and grid_matrix[i][j] in ['S']:
          group = []
          dfs(i, j, group)
          if group:
            groups.append(group)

    return groups


  def has_nearby_green(matrix, row, col, max_distance=3):
    rows = len(matrix)
    cols = len(matrix[0])

    for i in range(max(row - max_distance, 0), min(row + max_distance + 1, rows)):
        for j in range(max(col - max_distance, 0), min(col + max_distance + 1, cols)):
            if matrix[i][j] == 'G':
                return True

    return False

  def search_for_nearby_green(matrix):
    counter = 0; #counts how many Rs have green space
    rows = len(matrix)
    cols = len(matrix[0])

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 'R':
                if CityLayout.has_nearby_green(matrix, i, j, 3):
                    #print(f"Found nearby green for 'R' at ({i}, {j})")
                    counter = counter + 1
                #else:
                    #print(f"No nearby green for 'R' at ({i}, {j})")

    return counter


  def commercial_weight(layout): #Cs should be between 10% and 20%
      layout_matrix = [layout[i:i+width] for i in range(0, gridsize, width)]
      commercial_count = 0
      for i in range(height):
          for j in range(width):
            # Current tile
            tile = layout_matrix[i][j]
            if tile == 'C':
              commercial_count = commercial_count + 1
      commercial_weight = 1;
      if commercial_count < width * height * 0.1:
        commercial_weight = (commercial_count / gridsize) * 5 + 0.5
      elif commercial_count > width * height * 0.2:
        commercial_weight = (commercial_count / gridsize) * (-0.625) + 1.125
      return commercial_weight

  def green_weight(layout): #green tiles should be between 15% and 25%
      layout_matrix = [layout[i:i+width] for i in range(0, gridsize, width)]
      green_count = 0
      for i in range(height):
          for j in range(width):
            # Current tile
            tile = layout_matrix[i][j]
            if tile == 'G':
              green_count = green_count + 1
      green_weight = 1;
      if green_count < width * height * 0.15:
        green_weight = (green_count / gridsize) * 3.33 + 0.5
      elif green_count > width * height * 0.25:
        green_weight = (green_count / gridsize) * (-0.66) + 1.16

      return green_weight

  def street_weight(layout): #street tiles should be between 20% and 30%
      layout_matrix = [layout[i:i+width] for i in range(0, gridsize, width)]
      street_count = 0
      for i in range(height):
          for j in range(width):
            # Current tile
            tile = layout_matrix[i][j]
            if tile == 'S':
              street_count = street_count + 1
      street_weight = 1;
      if street_count < width * height * 0.2:
        street_weight = (street_count / gridsize) * 2.5 + 0.5
      elif street_count > width * height * 0.3:
        street_weight = (street_count / gridsize) * (-0.714) + 1.214

      return street_weight

  def res_weight(layout): #R tiles should be between 25% and 40%
      layout_matrix = [layout[i:i+width] for i in range(0, gridsize, width)]
      res_count = 0
      for i in range(height):
          for j in range(width):
            # Current tile
            tile = layout_matrix[i][j]
            if tile == 'R':
              res_count = res_count + 1
      res_weight = 1;
      if res_count < width * height * 0.2:
        res_weight = (res_count / gridsize) * 2.5 + 0.5
      elif res_count > width * height * 0.3:
        green_weight = (res_count / gridsize) * (-0.714) + 1.214

      return res_weight

  def res_clusters_weight(layout):
      r_count = 0
      for i in layout:
        if i == 'R':
          r_count += 1
      res_weight = 1
      groups = CityLayout.find_groups(layout)

      groups_ok = 0
      sum = 0
      for i, group in enumerate(groups, 1):
        filtered_list = [item for item in group if item != True]
        num_elements = len(filtered_list)

        if num_elements >= 2 and num_elements <= 10:
          groups_ok += 1

      if groups_ok > len(groups) / 2:
        res_weight *= groups_ok / len(groups)
      else:
        res_weight = 0.5

      return res_weight

  def street_adjacency_weight(layout):
      groups = CityLayout.find_groups(layout)
      street_adj_weight = 1
      count = 0
      for group in groups:
        if True in group:
          count += 1
      #count = sum(True in group for group in groups)
      if len(groups) != 0:
       percentage = count / len(groups)
      else:
        percentage = 0
      #print("perceny is now: " + str(percentage))
      street_adj_weight = 0.5 * percentage + 0.5

      return street_adj_weight

  def nearby_green_weight(layout):
      layout_matrix = [layout[i:i+width] for i in range(0, gridsize, width)]
      counter = CityLayout.search_for_nearby_green(layout_matrix)
      r_count = 0
      for row in layout_matrix:
        for tile in row:
            if tile == 'R':
                r_count += 1
      if r_count != 0:
        nearby_green_weight = counter / r_count
      else:
        nearby_green_weight = 0.5

      return nearby_green_weight

  def street_connectivity_weight(layout):
      groups = CityLayout.find_streets_connected(layout)
      street_connectivity = 0
      for g in groups:
        if len(g) != 0:
          street_connectivity += 1/(len(g)**2)

      s_count = 0
      for i in layout:
        if i == 'S':
          s_count += 1

      if s_count != 1:
        return street_connectivity*(s_count*2)/(1-s_count * 3) - s_count * 3/(1-s_count*3)
      else:
        return 0.5

  def elev_weight_normal(layout, elev_grid):
      elevations_matrix = [elev_grid[i:i+width] for i in range(0, len(layout), width)]
      layout_matrix = [layout[i:i+width] for i in range(0, len(layout), width)]
      highest_point = 0
      lowest_point = 1500

      for row in elevations_matrix:
        for value in row:
          if value > highest_point:
            highest_point = value
          elif value < lowest_point:
            lowest_point = value
      elev_weight = 1

      for i in range(width):
        for j in range(height):
            tyle = layout_matrix[i][j]
            if tyle == 'R' and elevations_matrix[i][j] > (highest_point - lowest_point) / 3: #642
              elev_weight += elevations_matrix[i][j] * (1.5 / ((-2) * highest_point - lowest_point)) + 0.5 + highest_point * (1.5 / (2 * highest_point + lowest_point))
            elif tyle == 'C' and elevations_matrix[i][j] > (highest_point - lowest_point) / 5: #385
              elev_weight += elevations_matrix[i][j] * (2.5 / ((-4) * highest_point - lowest_point)) + 0.5 + highest_point * (2.5 / (4 * highest_point + lowest_point))
            elif tyle == 'G' and elevations_matrix[i][j] > (highest_point - lowest_point) / 3: #642
              elev_weight += elevations_matrix[i][j] * (1.5 / (2 * highest_point + lowest_point)) + 1 - 0.5 * (highest_point - lowest_point) / (2 * highest_point + lowest_point)
            else:
              elev_weight += 1
      return elev_weight/gridsize

  def calculate_fitness(layout, elev_grid):
      # Transform the linear string into a 2D array
      layout_matrix = [layout[i:i+width] for i in range(0, gridsize, width)]
      elevations_matrix = [elev_grid[i:i+width] for i in range(0, len(elev_grid), width)]
      fitness = 0

      fitness = fitness + 10 * CityLayout.elev_weight_normal(layout, elev_grid)

      fitness = fitness + 10 * CityLayout.commercial_weight(layout)

      fitness = fitness + 10 * CityLayout.green_weight(layout)

      fitness = fitness + 10 * CityLayout.res_weight(layout)

      fitness = fitness + 10 * CityLayout.street_weight(layout)

      fitness = fitness + 10 * CityLayout.res_clusters_weight(layout)

      fitness = fitness + 10 * CityLayout.street_adjacency_weight(layout)

      #verify if there is green space at a distance of maximum 3 tiles from R tiles
      fitness = fitness + 10 * CityLayout.nearby_green_weight(layout)

      #print(f"The weight for street connectivity is {street_connectivity_weight}")
      fitness = fitness + 10 * CityLayout.street_connectivity_weight(layout)

      return fitness


def distance(x, y):
  count = 0
  for item1, item2 in zip(x, y):
    if item1 != item2:
      count += 1

  return count

# city layout using Ant Colony Optimization

# Subclass to be used by the ACO class
class UrbanGardeningProblem:
    def __init__(self, elevations):
        self.elevations = elevations

    def evaluate(self, solution):
        # Evaluates the urban layout solution using the fitness function from CityLayout
        return CityLayout.calculate_fitness(solution, self.elevations)

# The class for testing
class ACO_UrbanGardening:
    def __init__(self, grid_size, num_ants, num_iterations, problem, evaporation_rate, alpha, beta, elevations_string):
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
        self.best_fitness_historic = []
        self.diversity_historic = []


    # Display the solution as a matrix
    def visualize_solution(self, solution):
        color_map = {'C': 'blue', 'R': 'brown', 'S': 'grey', 'G': 'green'}
        color_array = [color_map[tile] for tile in solution]

        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(width):
            for j in range(height):
                index = i * width + j
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color_array[index]))

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.invert_yaxis()
        ax.axis('off')
        plt.show()

    # Apply the algorithm
    def run(self):
        best_solution = None # No solution yet
        best_fitness = -np.inf # No fitness yet

        # Start of the algorithm
        for iteration in range(self.num_iterations):
            solutions = [self.generate_solution() for _ in range(self.num_ants)]
            fitnesses = [self.problem.evaluate(solution) for solution in solutions]

            self.record_history(solutions, fitnesses)  # Record the evolution history
            # Update pheromones globally
            self.update_global_pheromones(solutions, fitnesses)

            # Check for new best solution
            current_best_index = np.argmax(fitnesses)
            if fitnesses[current_best_index] > best_fitness:
                best_fitness = fitnesses[current_best_index]
                best_solution = solutions[current_best_index]

        self.visualize_solution(best_solution)
        return (best_solution, best_fitness, self.commercial_weight_history,
                self.green_weight_history, self.res_weight_history, self.street_adjacency_history,
                self.street_connectivity_history,
                self.elev_weight_overall_history)


    # Keep track of the history for comparison
    def record_history(self, solutions, fitnesses):
        self.history.append(solutions)
        best_index = np.argmax(fitnesses)
        best_solution = solutions[best_index]

        self.solution_history_fitness.append(CityLayout.calculate_fitness(best_solution, self.problem.elevations))
        self.commercial_weight_history.append(CityLayout.commercial_weight(best_solution))
        self.green_weight_history.append(CityLayout.green_weight(best_solution))
        self.res_weight_history.append(CityLayout.res_weight(best_solution))
        self.street_adjacency_history.append(CityLayout.street_adjacency_weight(best_solution))
        self.nearby_green_weight_history.append(CityLayout.nearby_green_weight(best_solution))
        self.street_connectivity_history.append(CityLayout.street_connectivity_weight(best_solution))
        self.elev_weight_overall_history.append(CityLayout.elev_weight_normal(best_solution, self.problem.elevations))
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

        grid_matrix = [solution[i:i + width] for i in range(0, len(solution), width)]
        i, j = divmod(position, width)

        # Neighborhood analysis with boundary checks
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E
        neighbors = [
            (i + di, j + dj)
            for di, dj in directions
            if 0 <= i + di < height and 0 <= j + dj < width
        ]

        # Filter neighbors that are within the bounds of the solution matrix
        valid_neighbors = [(ni, nj) for ni, nj in neighbors if 0 <= ni < height and 0 <= nj < width]

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

# Example usage
grid_size = 100
num_ants = 10
num_iterations = 15
elevations = [random.randint(0, 1000) for _ in range(grid_size)]
problem = UrbanGardeningProblem(elevations)

aco = ACO_UrbanGardening(grid_size, num_ants, num_iterations, problem, 0.1, 1, 1, elevations)
best_solution, best_fitness, commercial_weight_history, green_weight_history, res_weight_history, street_adjacency_history, street_connectivity_history, elev_weight_overall_history = aco.run()

print("Best Fitness:", best_fitness)
aco.visualize_fitness_history()
# Optional: Visualize the best solution
# aco.visualize_solution(best_solution)

"""
plt.figure()
plt.plot(commercial_weight_history, label='commercial weight')
plt.plot(green_weight_history, label='green weight')
plt.plot(res_weight_history, label='res weight')
plt.plot(street_adjacency_history, label='street adjacency')
#plt.plot(nearby_green_history, label='nearby green')
plt.plot(street_connectivity_history, label='street connectivity')
plt.plot(elev_weight_overall_history, label='elevations overall')
#plt.plot(solution_history, label='fitness')
plt.xlabel('Generation')
plt.ylabel('')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
"""