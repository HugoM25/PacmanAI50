import csv
import numpy as np

def load_level_from_csv(file_path):
    """
    Load a level from a CSV file.
    :param file_path: The path to the CSV file.
    :return: The level as a 2D numpy array of integers.
    """
    level = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            level.append([int(cell) for cell in row])
    return np.array(level)
