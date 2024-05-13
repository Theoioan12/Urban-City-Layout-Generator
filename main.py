# Imports for all of the code
import math
from random import Random
from time import time
from inspyred import ec, benchmarks
from matplotlib import pyplot as plt

# Static variables
gridsize = 25
height = 5
width = 5

import random
# Generate 16 random values between 0 and 1000
elevations_string = [random.randint(0, 1000) for _ in range(gridsize)]
#elevations_string = [948,924,1000,903,936,  935,  990,  939,   68   43   40   96   71  968
  #938   43   25   97   16   88  990  901   99   14   67   22    9  909
  #940   84    0   81   82   76  985  946   95   80    1   46   80  993
  #956  915  981  932  996  993  982]

# Convert the list of random values to a string

# Print the generated string
#print((elevations_string))

import numpy as np

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
#print(elevation_matrix)
elevations_string = elevation_matrix.flatten()
print(elevations_string)
#elevations_string = [850, 845, 830, 800, 750, 850, 845, 830, 800, 750, 850, 845, 830, 800, 750, 850, 845, 830, 800, 750, 850, 845, 830, 800, 750]


class CityLayout(benchmarks.Benchmark):
  def __init__(self, elevations_string):
        benchmarks.Benchmark.__init__(self, len(elevations_string))
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
            viol = CityLayout.street_adjacency_weight(c)**beta + CityLayout.commercial_weight(c)**beta + CityLayout.elev_weight_normal(c, self.elevations_string)**beta + CityLayout.green_weight(c)**beta + CityLayout.nearby_green_weight(c)**beta + CityLayout.res_weight(c)**beta + CityLayout.street_connectivity_weight(c)**beta
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
  """

  def search_s_groups(matrix):
    def dfs(row, col, size):
        # Check if the current cell is within the bounds of the matrix and is an 'S' tile
        if row < 0 or row >= len(matrix) or col < 0 or col >= len(matrix[0]) or matrix[row][col] != 'S':
            return size

        # Mark the current cell as visited by changing its value to something else ('X', for example)
        matrix[row][col] = 'X'
        size += 1

        # Explore the neighboring cells (up, down, left, right)
        size = dfs(row - 1, col, size)  # Up
        size = dfs(row + 1, col, size)  # Down
        size = dfs(row, col - 1, size)  # Left
        size = dfs(row, col + 1, size)  # Right

        return size

    max_group_size = 0

    # Iterate through the matrix
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 'S':
                # Start DFS from the current cell
                group_size = dfs(i, j, 0)
                max_group_size = max(max_group_size, group_size)

    return max_group_size

  """

  def commercial_weight(layout):
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
        commercial_weight = (commercial_count / (width * height)) * 10
      elif commercial_count > width * height * 0.3:
        commercial_weight = (commercial_count / (width * height)) * (-1.42) + 1.426
      return commercial_weight

  def green_weight(layout):
      layout_matrix = [layout[i:i+width] for i in range(0, gridsize, width)]
      green_count = 0
      for i in range(height):
          for j in range(width):
            # Current tile
            tile = layout_matrix[i][j]
            if tile == 'G':
              green_count = green_count + 1
      green_weight = 1;
      if green_count < width * height * 0.3:
        green_weight = (green_count / (width * height)) * 3.33
      elif green_count > width * height * 0.5:
        green_weight = (green_count / (width * height)) * (-2) + 2

      return green_weight

  def res_weight(layout):
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
        """
        dist = 0
        if num_elements < 2:
          dist = 1/((2 - r_count)**2)
        elif num_elements > 10:
          dist = 1/((r_count - 10)**2)
        sum += dist


        """

        if num_elements >= 2 and num_elements <= 10:
          groups_ok += 1

        """
        if num_elements < 2:
          res_weight = res_weight * 0.5 * num_elements
        elif num_elements > 10:
          res_weight = res_weight * (0.5 + 5 / gridsize)
          """
      if len(groups):
        res_weight *= groups_ok / len(groups)

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
      """
      layout_matrix = [layout[i:i+width] for i in range(0, gridsize, width)]
      matrix_copy = [row[:] for row in layout_matrix]
      biggest_street = CityLayout.search_s_groups(matrix_copy)
      #print(f"The biggest street is {biggest_street}")

      s_count = 0
      for row in layout_matrix:
        for tile in row:
            if tile == 'S':
                s_count += 1
      if s_count != 0:
        street_connectivity_weight = biggest_street / s_count
      else:
        street_connectivity_weight = 0.5
      """

      return street_connectivity*(s_count**2)/(1-s_count**3) - s_count**3/(1-s_count**3)

  def elev_weight_normal(layout, elev_grid):
      elevations_matrix = [elev_grid[i:i+width] for i in range(0, len(elev_grid), width)]
      layout_matrix = [layout[i:i+width] for i in range(0, len(layout), width)]
      highest_point = 0
      lowest_point = 1500

      for row in elevations_matrix:
        for value in row:
          if int(value) > int(highest_point):
            highest_point = int(value)
          elif value < lowest_point:
            lowest_point = value
      elev_weight = 1
      for i in range(width):
        for j in range(height):
            tyle = layout_matrix[i][j]
            if tyle == 'R' and elevations_matrix[i][j] > (highest_point - lowest_point) / 3: #642
              elev_weight += elevations_matrix[i][j] * (1.5 / ((-2)*highest_point - lowest_point)) + 0.5 + highest_point*(1.5 / (2 * highest_point + lowest_point))
            elif tyle == 'C' and elevations_matrix[i][j] > (highest_point - lowest_point) / 5: #385
              elev_weight += elevations_matrix[i][j] * (2.5 / ((-4)*highest_point - lowest_point)) + 0.5 + highest_point*(2.5 / (4 * highest_point + lowest_point))
            elif tyle == 'G' and elevations_matrix[i][j] > (highest_point - lowest_point) / 3: #642
              elev_weight += elevations_matrix[i][j] * (1.5 / (2 * highest_point + lowest_point)) + 1 - 0.5 * (highest_point - lowest_point) / (2 * highest_point + lowest_point)
            else:
              elev_weight += 1
      return elev_weight/gridsize

  def calculate_fitness(self, layout, elevations_string):
      # Transform the linear string into a 2D array
      layout_matrix = [layout[i:i+width] for i in range(0, gridsize, width)]
      elevations_matrix = [elevations_string[i:i+width] for i in range(0, len(elevations_string), width)]

      highest_point = 0
      lowest_point = 1500
      for row in elevations_matrix:
        for value in row:
          if value > highest_point:
            highest_point = value
          elif value < lowest_point:
            lowest_point = value

      fitness = 0
      green_count = 0
      # Iterate over each tile in the grid
      for i in range(height):
          for j in range(width):
            # Current tile
            tile = layout_matrix[i][j]
            score = 0
            weight = 1;
            if tile == 'G':
              green_count = green_count + 1

            # Check surrounding tiles within the 3x3 filter
            """
            for di in range(-1, 2):  # delta i
                for dj in range(-1, 2):  # delta j
                    ni, nj = i + di, j + dj  # neighbor indices
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor = layout_matrix[ni][nj]
                        # Apply rules specific to each tile type
                        if tile == 'R' and (neighbor == 'G' or neighbor == 'S'):
                            score += 2
                        elif tile == 'C' and (neighbor == 'R' or neighbor == 'S'):
                            score += 1
                        elif tile == 'G' and neighbor == 'R':
                            score += 3
                        elif tile == 'S' and (neighbor == 'R' or neighbor == 'C'):
                            score += 4

            """
            """
            elev_weight = 1
            if tile == 'R' and elevations_matrix[i][j] > (highest_point - lowest_point) / 3: #642
              elev_weight = elevations_matrix[i][j] * (1.5 / ((-2)*highest_point - lowest_point)) + 0.5 + highest_point*(1.5 / (2 * highest_point + lowest_point))
            elif tile == 'C' and elevations_matrix[i][j] > (highest_point - lowest_point) / 5: #385
              elev_weight = elevations_matrix[i][j] * (2.5 / ((-4)*highest_point - lowest_point)) + 0.5 + highest_point*(2.5 / (4 * highest_point + lowest_point))
            elif tile == 'G' and elevations_matrix[i][j] > (highest_point - lowest_point) / 3: #642
              elev_weight = elevations_matrix[i][j] * (1.5 / (2 * highest_point + lowest_point)) + 1 - 0.5 * (highest_point - lowest_point) / (2 * highest_point + lowest_point)

            score = score + 10 * elev_weight
            """
            fitness += score

            #print(tile + ": " + str(fitness))
      fitness = fitness + 10 * CityLayout.elev_weight_normal(layout, self.elevations_string)

      fitness = fitness + 10 * CityLayout.commercial_weight(layout)

      fitness = fitness + 10 * CityLayout.green_weight(layout)

      fitness = fitness + 10 * CityLayout.res_weight(layout)

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

class ACO_CityLayout:
    def __init__(self, grid_size, num_ants, num_iterations, problem):
        self.grid_size = grid_size
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.problem = problem
        self.pheromones = np.ones((grid_size * grid_size, 4))  # Pheromone levels for each tile type at each grid position
        self.evaporation_rate = 0.1
        self.alpha = 3  # Pheromone influence
        self.beta = 1   # Heuristic influence (not used here)

        # Historical tracking
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

    def run(self):
        best_solution = None
        best_fitness = -np.inf
        for _ in range(self.num_iterations):
            solutions = [self.generate_solution() for _ in range(self.num_ants)]
            fitnesses = [self.problem.calculate_fitness(solution, elevations_string) for solution in solutions]
            # Assuming update_pheromones and other methods exist
            best_index = np.argmax(fitnesses)
            if fitnesses[best_index] > best_fitness:
                best_fitness = fitnesses[best_index]
                best_solution = solutions[best_index]

            self.update_pheromones(solutions, fitnesses)
            self.record_history(solutions, fitnesses)

        self.visualize_solution(best_solution)
        return best_solution, best_fitness

    def generate_solution(self):
        solution = []
        tile_types = ['R', 'C', 'S', 'G']
        for i in range(self.grid_size * self.grid_size):
            probabilities = np.power(self.pheromones[i], self.alpha)
            tile = random.choices(tile_types, weights=probabilities / probabilities.sum())[0]
            solution.append(tile)
        return solution

    def update_pheromones(self, solutions, fitnesses):
        for solution, fitness in zip(solutions, fitnesses):
            for i, tile in enumerate(solution):
                tile_index = ['R', 'C', 'S', 'G'].index(tile)
                self.pheromones[i, tile_index] += fitness
        self.pheromones *= (1 - self.evaporation_rate)

    def record_history(self, solutions, fitnesses):
        """
      self.history.append(solutions)
      best_index = np.argmax(fitnesses)
      best_solution = solutions[best_index]

      self.solution_history_fitness.append(self.problem.calculate_fitness(best_solution, elevations_string))
      self.commercial_weight_history.append(self.problem.commercial_weight(best_solution))
      self.green_weight_history.append(self.problem.green_weight(best_solution))
      self.res_weight_history.append(self.problem.res_weight(best_solution))
      self.street_adjacency_history.append(self.problem.street_adjacency_weight(best_solution))
      self.nearby_green_weight_history.append(self.problem.nearby_green_weight(best_solution))
      self.street_connectivity_history.append(self.problem.street_connectivity_weight(best_solution))
      self.elev_weight_overall_history.append(self.problem.elev_weight_normal(best_solution))
      self.best_fitness_historic.append(fitnesses[best_index])
      self.diversity_historic.append(self.calculate_diversity(solutions))
"""

    def calculate_diversity(self, solutions):
        value_map = {'R': 0, 'S': 1, 'C': 2, 'G': 3}
        numerical_solutions = [[value_map[tile] for tile in solution] for solution in solutions]
        return np.std(numerical_solutions, axis=0).mean()

    def visualize_solution(self, solution):
        color_map = {'C': 'blue', 'R': 'brown', 'S': 'grey', 'G': 'green'}
        color_array = [color_map[tile] for tile in solution]

        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                index = i * self.grid_size + j
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color_array[index]))

        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.invert_yaxis()
        ax.axis('off')
        plt.show()

# Usage Example
elevations_string = [random.randint(0, 1000) for _ in range(gridsize)]
problem = CityLayout(elevations_string)
aco = ACO_CityLayout(gridsize, 10,100, problem)
best_solution, best_fitness = aco.run()
