import os
import pygame
import copy
import json

# Constants -------------------------------------------
# Define the colors
COLORS = {
    'BLACK': (0, 0, 0),
    'WHITE': (255, 255, 255),
    'RED': (255, 0, 0),
    'GREEN': (0, 255, 0),
    'BLUE': (0, 0, 255),
    'LIGHT_GRAY': (200, 200, 200),
    'YELLOW': (255, 255, 0),
    'DARK_GRAY': (50, 50, 50),
    'DARKER_GRAY': (30, 30, 30),
    'CHECKER_GRAY': (128, 128, 128),
    'CHECKER_DARK_GRAY': (100, 100, 100),
}
# Set the dimensions of the grid
GRID_WIDTH_SIZE = 28
GRID_HEIGHT_SIZE = 31
CELL_SIZE = 30
GRID_WIDTH = GRID_WIDTH_SIZE * CELL_SIZE
GRID_HEIGHT = GRID_HEIGHT_SIZE * CELL_SIZE

# DISPLAY SETTINGS
MAX_TILES_PER_ROW = 3

# Class of LevelEditor -------------------------------------------
class LevelEditor:
    def __init__(self) -> None:
        # Initialize Pygame
        pygame.init()

        # Create the grid
        self.grid = [[0] * GRID_WIDTH_SIZE for _ in range(GRID_HEIGHT_SIZE)]

        # Create the screen
        self.screen = pygame.display.set_mode((GRID_WIDTH + 100, GRID_HEIGHT + 50))
        pygame.display.set_caption("Level Editor")

        # Create the font
        self.font = pygame.font.Font(None, 24)

        # Load the tileset used to draw the grid
        self.load_tileset_to_paint("tileset.png")

        # Create the buttons
        self.save_button = pygame.Rect(10, GRID_HEIGHT + 10, 80, 30)
        self.reset_button = pygame.Rect(100, GRID_HEIGHT + 10, 80, 30)
        self.mirror_horizontal_button = pygame.Rect(320, GRID_HEIGHT + 10, 120, 30)
        self.mirror_vertical_button = pygame.Rect(450, GRID_HEIGHT + 10, 120, 30)

        # Track the selected tile
        self.selected_tile = (0, 0)

        # Track the initial state when the mouse button is first pressed
        self.initial_state = None
        self.initial_tile = None

        # Track the history of grid states for undo and redo functionality
        self.history = []
        self.future = []

        self.running = True

    def load_tileset_to_paint(self, tileset_path):
        #Load the json file with the tileset
        self.tileset_data = None
        with open(os.path.join("pacman_game/res", "tileset_pacman1.json"), "r") as f:
            self.tileset_data = json.load(f)["tileset"]
        self.tile_size = self.tileset_data["tile_size"]

        # Load the tileset used to draw the grid
        self.tileset = pygame.image.load(os.path.join("pacman_game/res", tileset_path)).convert_alpha()
        self.tileset_columns = self.tileset.get_width() // self.tile_size
        self.tileset_rows = self.tileset.get_height() // self.tile_size


        # Fill a list with the tiles from the tileset
        count = 0
        self.tiles = []
        for i in range(self.tileset_rows):
            for j in range(self.tileset_columns):
                if count >= self.tileset_data["tile_count"]:
                    break
                tile = (j, i)
                self.tiles.append(tile)
                count += 1

    def save_state(self):
        '''
        Save the current state of the grid to the history list
        (used for undo and redo functionality)
        '''
        self.history.append(copy.deepcopy(self.grid))
        self.future.clear()

    def draw_tile(self, window, tile_x, tile_y, pos_x, pos_y, cell_size):
        '''
        Draw a tile from the tileset on the screen
        :param window: the Pygame window to draw on
        :param tile_x: the x-coordinate of the tile in the tileset
        :param tile_y: the y-coordinate of the tile in the tileset
        :param pos_x: the x-coordinate to draw the tile on the screen
        :param pos_y: the y-coordinate to draw the tile on the screen
        :param cell_size: the size of the tile to draw
        '''
        tile_rect = pygame.Rect(tile_x * self.tile_size, tile_y * self.tile_size, self.tile_size, self.tile_size)
        tile_image = self.tileset.subsurface(tile_rect)
        tile_image = pygame.transform.scale(tile_image, (cell_size, cell_size))
        window.blit(tile_image, (pos_x, pos_y))

    def flood_fill(self, x, y, target_tile, replacement_tile):
        '''
        Flood fill algorithm to fill an area with the selected tile
        :param x: the x-coordinate of the starting cell
        :param y: the y-coordinate of the starting cell
        :param target_tile: the tile to be replaced
        :param replacement_tile: the tile to replace with
        '''
        if target_tile == replacement_tile:
            return
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if self.grid[cy][cx] == target_tile:
                self.grid[cy][cx] = replacement_tile
                if cx > 0:
                    stack.append((cx - 1, cy))
                if cx < GRID_WIDTH_SIZE - 1:
                    stack.append((cx + 1, cy))
                if cy > 0:
                    stack.append((cx, cy - 1))
                if cy < GRID_HEIGHT_SIZE - 1:
                    stack.append((cx, cy + 1))

    def run(self):
        '''
        Main loop of the LevelEditor
        '''
        while self.running:

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    # Handle undo functionality
                    if event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        if self.history:
                            self.future.append(copy.deepcopy(self.grid))
                            self.grid = self.history.pop()
                    # Handle redo functionality
                    elif event.key == pygame.K_y and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        if self.future:
                            self.history.append(copy.deepcopy(self.grid))
                            self.grid = self.future.pop()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Get the position of the mouse click (check if it's inside the grid or the tileset selection)
                    x, y = pygame.mouse.get_pos()

                    # Handle tile placement ---------------------------------------
                    if x < GRID_WIDTH and y < GRID_HEIGHT:
                        x_cell = x // CELL_SIZE
                        y_cell = y // CELL_SIZE
                        self.save_state()

                        # Handle bucket tool ---------------------------------------
                        if event.button == 2:  # Middle mouse button
                            target_tile = self.grid[y_cell][x_cell]
                            self.flood_fill(x_cell, y_cell, target_tile, self.selected_tile)
                        elif event.button == 3:  # Right mouse button
                            # Remove the tile
                            self.grid[y_cell][x_cell] = 0
                        else:
                            # Update the grid with the selected tile
                            self.grid[y_cell][x_cell] = self.selected_tile
                            self.initial_state = self.grid[y_cell][x_cell]
                            self.initial_tile = None
                            self.mouse_down = True  # Track mouse button state

                    # Handle tile selection ---------------------------------------
                    if x > GRID_WIDTH and y < GRID_HEIGHT:
                        max_size = self.screen.get_width() - GRID_WIDTH
                        size_tile = max_size // MAX_TILES_PER_ROW

                        column = (x - GRID_WIDTH) // size_tile
                        row = y // size_tile

                        index_tile_selected = row * MAX_TILES_PER_ROW + column

                        if index_tile_selected < len(self.tiles):
                            self.selected_tile = self.tiles[index_tile_selected]

                    # Handle button clicks ---------------------------------------
                    if y > GRID_HEIGHT :
                        if self.save_button.collidepoint(x, y):
                            print("Saving level...")
                            with open("level_data.csv", "w") as file:
                                max_digits = len(str(len(self.tiles)))
                                for row in self.grid:
                                    file.write(",".join(f"{self.tiles.index(tile):0{max_digits}d}" if tile != 0 else "0" * max_digits for tile in row) + "\n")

                        elif self.reset_button.collidepoint(x, y):
                            print("Resetting level...")
                            self.grid = [[0] * GRID_WIDTH_SIZE for _ in range(GRID_HEIGHT_SIZE)]
                        elif self.mirror_horizontal_button.collidepoint(x, y):
                            print("Mirroring horizontally...")
                            # Copy the left half of the grid to the right half
                            for i in range(GRID_HEIGHT_SIZE):
                                for j in range(GRID_WIDTH_SIZE // 2):
                                    self.grid[i][GRID_WIDTH_SIZE - j - 1] = self.grid[i][j]
                        elif self.mirror_vertical_button.collidepoint(x, y):
                            print("Mirroring vertically...")
                            # Copy the top half of the grid to the bottom half
                            for i in range(GRID_HEIGHT_SIZE // 2):
                                for j in range(GRID_WIDTH_SIZE):
                                    self.grid[GRID_HEIGHT_SIZE - i - 1][j] = self.grid[i][j]
                    # ----------------------------------------------------------------

                elif event.type == pygame.MOUSEBUTTONUP:
                    self.mouse_down = False  # Reset mouse button state

                elif event.type == pygame.MOUSEMOTION and getattr(self, 'mouse_down', False):
                    # Get the position of the mouse motion (check if it's inside the grid)
                    x, y = pygame.mouse.get_pos()
                    if x < GRID_WIDTH and y < GRID_HEIGHT:
                        x_cell = x // CELL_SIZE
                        y_cell = y // CELL_SIZE
                        # Update the grid with the selected tile
                        self.grid[y_cell][x_cell] = self.selected_tile

            # Fill the screen with color ---------------------------------------
            self.screen.fill(COLORS['BLACK'])

            # Draw the tiles options on the right side of the screen -----------
            available_width = self.screen.get_width() - GRID_WIDTH
            enlarged_tile_size = available_width // MAX_TILES_PER_ROW
            _row = 0
            _col = 0
            for i in range(len(self.tiles)):
                if i % MAX_TILES_PER_ROW == 0 and i != 0:
                    _row += 1
                    _col = 0
                tile_x, tile_y = self.tiles[i]
                pos_x = GRID_WIDTH + _col * enlarged_tile_size
                pos_y = _row * enlarged_tile_size

                # Draw the tile
                self.draw_tile(self.screen, tile_x, tile_y, pos_x, pos_y, enlarged_tile_size)
                # Add outline to the tiles
                if (tile_x, tile_y) == self.selected_tile:
                    pygame.draw.rect(self.screen, COLORS['WHITE'], (pos_x, pos_y, enlarged_tile_size, enlarged_tile_size), 2)
                else:
                    pygame.draw.rect(self.screen, COLORS['DARK_GRAY'], (pos_x, pos_y, enlarged_tile_size, enlarged_tile_size), 1)

                _col += 1
            # ----------------------------------------------------------------

            # Draw the grid checker background
            for i in range(GRID_HEIGHT_SIZE):
                for j in range(GRID_WIDTH_SIZE):
                    if self.grid[i][j] != 0:
                        self.draw_tile(self.screen, self.grid[i][j][0], self.grid[i][j][1], j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE)
                    else:
                        if (i + j) % 2 == 0:
                            pygame.draw.rect(self.screen, COLORS['DARK_GRAY'], (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                        else:
                            pygame.draw.rect(self.screen, COLORS['DARKER_GRAY'], (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            # ----------------------------------------------------------------

            # Draw the buttons -----------------------------------------------
            pygame.draw.rect(self.screen, COLORS['GREEN'], self.save_button)
            pygame.draw.rect(self.screen, COLORS['RED'], self.reset_button)
            pygame.draw.rect(self.screen, COLORS['WHITE'], self.mirror_horizontal_button)
            pygame.draw.rect(self.screen, COLORS['WHITE'], self.mirror_vertical_button)
            # Draw the text on the buttons
            text = self.font.render("Save", True, COLORS['WHITE'])
            self.screen.blit(text, (self.save_button.x + 10, self.save_button.y + 10))
            text = self.font.render("Reset", True, COLORS['WHITE'])
            self.screen.blit(text, (self.reset_button.x + 10, self.reset_button.y + 10))
            text = self.font.render("Mirror X", True, COLORS['BLACK'])
            self.screen.blit(text, (self.mirror_horizontal_button.x + 10, self.mirror_horizontal_button.y + 10))
            text = self.font.render("Mirror Y", True, COLORS['BLACK'])
            self.screen.blit(text, (self.mirror_vertical_button.x + 10, self.mirror_vertical_button.y + 10))
            # ----------------------------------------------------------------

            pygame.display.flip()

if __name__ == "__main__":
    level_editor = LevelEditor()
    level_editor.run()