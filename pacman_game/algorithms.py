import numpy as np
import heapq
from pacman_game.map import *
from pacman_game.constants import *


class NavigationAlgo:
    def __init__(self, current_map: np.array):
        self.current_map = current_map

    def heuristic(self, a, b):
        '''
        Function used to calculate the Manhattan distance between two points.
        '''
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_shortest_path_a_star(self, start_position, end_position):
        '''
        Find the shortest path between two points using the A* algorithm.
        '''
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        close_set = set()
        came_from = {}
        gscore = {tuple(start_position): 0}
        fscore = {tuple(start_position): self.heuristic(start_position, end_position)}
        oheap = []

        heapq.heappush(oheap, (fscore[tuple(start_position)], tuple(start_position)))

        while oheap:
            current = heapq.heappop(oheap)[1]

            if np.array_equal(current, end_position):
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                return data[::-1]  # Return reversed path

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                tentative_g_score = gscore[current] + 1

                if 0 <= neighbor[0] < self.current_map.shape[0]:
                    if 0 <= neighbor[1] < self.current_map.shape[1]:
                        if self.current_map[neighbor[0]][neighbor[1]] == WALL:
                            continue
                    else:
                        # Out of bounds
                        continue
                else:
                    # Out of bounds
                    continue

                neighbor = tuple(neighbor)
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue

                if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, end_position)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))

        return False  # No path found