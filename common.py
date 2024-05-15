# Imports for all of the code
from inspyred import ec, benchmarks
import random
import numpy as np
import matplotlib.pyplot as plt

"""
Modified the code from the generational
since I opted for an automatic testing.
"""
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

# city layout to be used for generation
"""
To this class I added width and height 
separately not as global variables.
"""
class CityLayout(benchmarks.Benchmark):
  # init method
  def __init__(self, elevations_string, width, height):
      benchmarks.Benchmark.__init__(self, len(elevations_string))
      self.elevations_string = elevations_string
      self.width = width
      self.height = height
      self.gridsize = width * height
      self.bounder = ec.DiscreteBounder(['R', 'C', 'S', 'G'])
      self.maximize = True
      self.best_all = None
      self.best_feasible = None

  def generator(self, random, args):
      chars = ['R', 'C', 'S', 'G']
      layout = [random.choice(chars) for _ in range(self.gridsize)]
      return layout

  def evaluator(self, candidates, args):
      fitness = []
      for candidate in candidates:
          fitness.append(self.calculate_fitness(candidate, self.elevations_string))
      return fitness

# The actual fitness function
  def calculate_fitness(self, layout, elev_grid):
      fitness = 0
      fitness += 10 * self.elev_weight_normal(layout, elev_grid)
      fitness += 10 * self.commercial_weight(layout)
      fitness += 10 * self.green_weight(layout)
      fitness += 10 * self.res_weight(layout)
      fitness += 10 * self.street_weight(layout)
      fitness += 10 * self.res_clusters_weight(layout)
      fitness += 10 * self.street_adjacency_weight(layout)
      fitness += 10 * self.nearby_green_weight(layout)
      fitness += 10 * self.street_connectivity_weight(layout)
      return fitness

  def commercial_weight(self, layout):  # Cs should be between 10% and 20%
      layout_matrix = [layout[i:i + self.width] for i in range(0, self.gridsize, self.width)]
      commercial_count = sum(1 for row in layout_matrix for tile in row if tile == 'C')
      commercial_weight = 1
      if commercial_count < self.width * self.height * 0.1:
          commercial_weight = (commercial_count / self.gridsize) * 5 + 0.5
      elif commercial_count > self.width * self.height * 0.2:
          commercial_weight = (commercial_count / self.gridsize) * (-0.625) + 1.125
      return commercial_weight

  def green_weight(self, layout):  # Green tiles should be between 15% and 25%
      layout_matrix = [layout[i:i + self.width] for i in range(0, self.gridsize, self.width)]
      green_count = sum(1 for row in layout_matrix for tile in row if tile == 'G')
      green_weight = 1
      if green_count < self.width * self.height * 0.15:
          green_weight = (green_count / self.gridsize) * 3.33 + 0.5
      elif green_count > self.width * self.height * 0.25:
          green_weight = (green_count / self.gridsize) * (-0.66) + 1.16
      return green_weight

  def street_weight(self, layout):  # Street tiles should be between 20% and 30%
      layout_matrix = [layout[i:i + self.width] for i in range(0, self.gridsize, self.width)]
      street_count = sum(1 for row in layout_matrix for tile in row if tile == 'S')
      street_weight = 1
      if street_count < self.width * self.height * 0.2:
          street_weight = (street_count / self.gridsize) * 2.5 + 0.5
      elif street_count > self.width * self.height * 0.3:
          street_weight = (street_count / self.gridsize) * (-0.714) + 1.214
      return street_weight

  def res_weight(self, layout):  # R tiles should be between 20% and 30%
      layout_matrix = [layout[i:i + self.width] for i in range(0, self.gridsize, self.width)]
      res_count = sum(1 for row in layout_matrix for tile in row if tile == 'R')
      res_weight = 1
      if res_count < self.width * self.height * 0.2:
          res_weight = (res_count / self.gridsize) * 2.5 + 0.5
      elif res_count > self.width * self.height * 0.3:
          res_weight = (res_count / self.gridsize) * (-0.714) + 1.214
      return res_weight

  def res_clusters_weight(self, layout):
      r_count = layout.count('R')
      res_weight = 1
      groups = self.find_groups(layout)
      groups_ok = sum(1 for group in groups if 2 <= len([item for item in group if item != True]) <= 10)

      if groups_ok > len(groups) / 2:
          res_weight *= groups_ok / len(groups)
      else:
          res_weight = 0.5

      return res_weight

  def street_adjacency_weight(self, layout):
      groups = self.find_groups(layout)
      count = sum(True in group for group in groups)
      percentage = count / len(groups) if groups else 0
      street_adj_weight = 0.5 * percentage + 0.5
      return street_adj_weight

  def nearby_green_weight(self, layout):
      layout_matrix = [layout[i:i + self.width] for i in range(0, self.gridsize, self.width)]
      counter = self.search_for_nearby_green(layout_matrix)
      r_count = sum(1 for row in layout_matrix for tile in row if tile == 'R')
      nearby_green_weight = counter / r_count if r_count else 0.5
      return nearby_green_weight

  def street_connectivity_weight(self, layout):
      groups = self.find_streets_connected(layout)
      street_connectivity = sum(1 / (len(g) ** 2) for g in groups if len(g) != 0)

      s_count = layout.count('S')
      if s_count != 1:
          return street_connectivity * (s_count * 2) / (1 - s_count * 3) - s_count * 3 / (1 - s_count * 3)
      else:
          return 0.5

  def elev_weight_normal(self, layout, elev_grid):
      elevations_matrix = [elev_grid[i:i + self.width] for i in range(0, len(layout), self.width)]
      layout_matrix = [layout[i:i + self.width] for i in range(0, len(layout), self.width)]
      highest_point = max(max(row) for row in elevations_matrix)
      lowest_point = min(min(row) for row in elevations_matrix)

      elev_weight = 1
      for i in range(self.width):
          for j in range(self.height):
              tile = layout_matrix[i][j]
              elevation = elevations_matrix[i][j]
              if tile == 'R' and elevation > (highest_point - lowest_point) / 3:
                  elev_weight += elevation * (1.5 / ((-2) * highest_point - lowest_point)) + 0.5 + highest_point * (
                              1.5 / (2 * highest_point + lowest_point))
              elif tile == 'C' and elevation > (highest_point - lowest_point) / 5:
                  elev_weight += elevation * (2.5 / ((-4) * highest_point - lowest_point)) + 0.5 + highest_point * (
                              2.5 / (4 * highest_point + lowest_point))
              elif tile == 'G' and elevation > (highest_point - lowest_point) / 3:
                  elev_weight += elevation * (1.5 / (2 * highest_point + lowest_point)) + 1 - 0.5 * (
                              highest_point - lowest_point) / (2 * highest_point + lowest_point)
              else:
                  elev_weight += 1

      return elev_weight / self.gridsize

  def find_groups(self, grid):
      def dfs(i, j, group):
          if 0 <= i < self.height and 0 <= j < self.width and visited[i][j] == False and grid_matrix[i][j] in ['R']:
              visited[i][j] = True
              group.append((i, j))
              for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                  dfs(i + di, j + dj, group)
          elif 0 <= i < self.height and 0 <= j < self.width and grid_matrix[i][j] == 'S':
              group.append(True)

      grid_matrix = [grid[i:i + self.width] for i in range(0, len(grid), self.width)]
      visited = [[False] * self.width for _ in range(self.height)]
      groups = []

      for i in range(self.height):
          for j in range(self.width):
              if visited[i][j] == False and grid_matrix[i][j] in ['R']:
                  group = []
                  dfs(i, j, group)
                  if group:
                      groups.append(group)

      return groups

  def find_streets_connected(self, grid):
      def dfs(i, j, group):
          if 0 <= i < self.height and 0 <= j < self.width and visited[i][j] == False and grid_matrix[i][j] in ['S']:
              visited[i][j] = True
              group.append((i, j))
              for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                  dfs(i + di, j + dj, group)

      grid_matrix = [grid[i:i + self.width] for i in range(0, len(grid), self.width)]
      visited = [[False] * self.width for _ in range(self.height)]
      groups = []

      for i in range(self.height):
          for j in range(self.width):
              if visited[i][j] == False and grid_matrix[i][j] in ['S']:
                  group = []
                  dfs(i, j, group)
                  if group:
                      groups.append(group)

      return groups

  def has_nearby_green(self, matrix, row, col, max_distance=3):
      rows = len(matrix)
      cols = len(matrix[0])

      for i in range(max(row - max_distance, 0), min(row + max_distance + 1, rows)):
          for j in range(max(col - max_distance, 0), min(col + max_distance + 1, cols)):
              if matrix[i][j] == 'G':
                  return True

      return False

  def search_for_nearby_green(self, matrix):
      counter = 0
      rows = len(matrix)
      cols = len(matrix[0])

      for i in range(rows):
          for j in range(cols):
              if matrix[i][j] == 'R':
                  if self.has_nearby_green(matrix, i, j, 3):
                      counter += 1

      return counter


def distance(x, y):
  count = 0
  for item1, item2 in zip(x, y):
    if item1 != item2:
      count += 1

  return count

"""
    the ACO implementation.
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
        self.best_fitness_historic = []
        self.diversity_historic = []
        self.city_layout = CityLayout(self.elevations_string, self.width, self.height)


    # Display the solution as a matrix
    def visualize_solution(self, solution):
        color_map = {'C': 'blue', 'R': 'brown', 'S': 'grey', 'G': 'green'}
        color_array = [color_map[tile] for tile in solution]

        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(self.width):
            for j in range(self.height):
                index = i * self.width + j
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color_array[index]))

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
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

        self.solution_history_fitness.append(self.city_layout.calculate_fitness(best_solution, self.problem.elevations))
        self.commercial_weight_history.append(self.city_layout.commercial_weight(best_solution))
        self.green_weight_history.append(self.city_layout.green_weight(best_solution))
        self.res_weight_history.append(self.city_layout.res_weight(best_solution))
        self.street_adjacency_history.append(self.city_layout.street_adjacency_weight(best_solution))
        self.nearby_green_weight_history.append(self.city_layout.nearby_green_weight(best_solution))
        self.street_connectivity_history.append(self.city_layout.street_connectivity_weight(best_solution))
        self.elev_weight_overall_history.append(self.city_layout.elev_weight_normal(best_solution, self.problem.elevations))
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
"""
# Example usage
grid_size = 100
height = 10
width = 10
num_ants = 10
num_iterations = 15
elevations = [random.randint(0, 1000) for _ in range(grid_size)]
problem = UrbanGardeningProblem(elevations, width, height)

aco = ACO_UrbanGardening(grid_size, num_ants, num_iterations, problem, 0.1, 1, 1, elevations, width, height)
best_solution, best_fitness, commercial_weight_history, green_weight_history, res_weight_history, street_adjacency_history, street_connectivity_history, elev_weight_overall_history = aco.run()

print("Best Fitness:", best_fitness)
aco.visualize_fitness_history()
# Optional: Visualize the best solution
# aco.visualize_solution(best_solution)
"""
