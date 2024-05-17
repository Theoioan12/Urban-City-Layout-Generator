"""
 Imported code but modified by Buliga Theodor Ioan
 UPM ETSISI - Bioinspired Algorithms for Optimization 2023-2024
"""

from inspyred import ec, benchmarks
import random
import numpy as np
import matplotlib.pyplot as plt

"""
 Modified the code from the generational
 since I opted for an automatic testing.
 
 !No algorithmical changes, only implementing 
 details so we can facilitate the testing!
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
 To this class I added width and height as atributes 
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