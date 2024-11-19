import heapq
import numpy as np

# Directions possibles : haut, bas, gauche, droite
DIRECTIONS = [
    (-1, 0),  # haut
    (1, 0),  # bas
    (0, -1),  # gauche
    (0, 1)  # droite
]


def heuristic(a, b):
    # Heuristique de Manhattan
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(map_data, start, goal):
    """
    Implémente l'algorithme A* pour trouver le chemin optimal entre start et goal.

    :param map_data: La carte sous forme de matrice numpy
    :param start: Position de départ (tuple)
    :param goal: Position cible (tuple)
    :return: Liste des mouvements pour atteindre la cible
    """
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if np.array_equal(current, goal):
            return reconstruct_path(came_from, current)

        for direction in DIRECTIONS:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            if not (0 <= neighbor[0] < map_data.shape[0] and 0 <= neighbor[1] < map_data.shape[1]):
                continue  # En dehors de la carte
            if map_data[neighbor[0], neighbor[1]] == 9:  # 9 représente un mur
                continue  # Position invalide (mur)

            tentative_g_score = g_score[current] + 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # Retourne un chemin vide si aucune solution n'est trouvée


def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()  # On inverse le chemin pour partir du début
    return path


def choose_pacman_to_chase(pacman_positions, blinky_position, map_data, detection_range):
    """
    Choisir le PACMAN à poursuivre basé sur la distance et la visibilité à l'aide de A*.

    :param pacman_positions: Liste des positions des PACMANS
    :param blinky_position: Position actuelle de Blinky
    :param map_data: La carte sous forme de matrice numpy
    :param detection_range: Portée maximale de détection
    :return: La position du PACMAN à poursuivre ou None s'il n'y en a aucun à portée
    """
    closest_pacman = None
    min_distance = float('inf')

    for pacman_position in pacman_positions:
        # Calcul de la distance de Manhattan
        distance = abs(blinky_position[0] - pacman_position[0]) + abs(blinky_position[1] - pacman_position[1])

        # Si la distance est dans la portée de détection, vérifier si Pacman est accessible
        if distance <= detection_range:
            # Utiliser A* pour vérifier si Pacman est accessible
            path = astar(map_data, blinky_position, pacman_position)

            if len(path) > 0 and distance < min_distance:
                closest_pacman = pacman_position
                min_distance = distance

    return closest_pacman

