from pacman_game import PacmanEnv
from human_collector import HumanCollector

if __name__ == "__main__":
    # Initialize the environment
    environment = PacmanEnv(["pacman_game/res/levels/level_1.csv"])

    # Initialize human collector
    human_collector = HumanCollector(environment, "human_collector_data.json", iterations=100)

    # Collect trajectories
    human_collector.collect_trajectories()




