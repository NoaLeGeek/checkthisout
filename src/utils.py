import os

import pygame

from src.config import config

def load_sound(filepath: str):
    """
    Loads a sound file and sets its volume based on the application's configuration.

    Parameters:
        filepath (str): The path to the sound file to be loaded.

    Returns:
        pygame.mixer.Sound: The loaded sound object with the volume set.
    """
    sound = pygame.mixer.Sound(filepath)
    sound.set_volume(config.volume)
    return sound 

def sign(x: int) -> int:
    """
    Determines the sign of an integer.

    This function returns 1 if the input integer is positive or zero, 
    and -1 if the input integer is negative.

    Parameters:
        x (int): The integer whose sign is to be determined.

    Returns:
        int: 1 if the input is non-negative, -1 if the input is negative.
    """
    return (x >= 0) - (x < 0)

def left_click() -> bool:
    """
    Checks if the left mouse button is currently being pressed.

    Returns:
        bool: True if the left mouse button is pressed, False otherwise.
    """
    return bool(pygame.mouse.get_pressed()[0])

def right_click() -> bool:
    """
    Check if the right mouse button is currently being pressed.

    This function uses Pygame's `mouse.get_pressed()` method to determine
    if the right mouse button (index 2) is pressed.

    Returns:
        bool: True if the right mouse button is pressed, False otherwise.
    """
    return bool(pygame.mouse.get_pressed()[2])

def get_value(flipped: int, white_value: int, black_value: int) -> int:
    """
    Determines and returns a value based on the flipped state.

    This function is useful for selecting between two values (e.g., for white 
    and black pieces in a chess game) depending on the flipped state.

    Parameters:
        flipped (int): Indicates the state, must be either 1 or -1. 
                       A value of 1 typically represents the "white" state, 
                       while -1 represents the "black" state.
        white_value (int): The value to return if flipped is 1.
        black_value (int): The value to return if flipped is -1.

    Returns:
        int: The selected value based on the flipped state.

    Raises:
        ValueError: If the flipped parameter is not -1 or 1.
    """
    if flipped not in (-1, 1):
        raise ValueError("flipped must be -1 or 1, not " + str(flipped))
    if flipped == 1:
        return white_value
    elif flipped == -1:
        return black_value

def get_pos(coord: tuple[int, int]) -> tuple[int, int]:
    """
    Converts screen coordinates to board coordinates based on the configuration settings.

    This function takes a tuple representing the x and y screen coordinates and calculates
    the corresponding row and column on the chessboard, considering margins, tile size, 
    and evaluation bar width.

    Parameters:
        coord (tuple[int, int]): A tuple containing the x and y screen coordinates.

    Returns:
        tuple[int, int]: A tuple containing the row and column indices on the chessboard.
    """
    x, y = coord
    return (y - config.margin) // config.tile_size, (x - config.margin - config.eval_bar_width) // config.tile_size

def flip_pos(pos: tuple[int, int] | int, flipped: int = -1) -> tuple[int, int] | int:
    """
    Flips the position(s) on a chessboard-like grid based on the `flipped` parameter.

    This function is useful for transforming positions when flipping the perspective
    of a chessboard, such as switching between white's and black's view.

    Parameters:
        pos (tuple[int, int] | int): The position(s) to flip. It can be a single integer
            or a tuple of two integers representing coordinates on the grid.
        flipped (int, optional): Determines whether to flip the position(s).
            - If `flipped` is 1, the position(s) remain unchanged.
            - If `flipped` is -1, the position(s) are flipped. Default is -1.

    Returns:
        tuple[int, int] | int: The flipped position(s). If the input `pos` is a single
        integer, the output will also be a single integer. If the input `pos` is a tuple,
        the output will be a tuple of flipped coordinates.
    """
    # No flip if flipped = 1
    # Flip if flipped = -1
    if isinstance(pos, int):
        pos = (pos,)
    flipped_pos = [get_value(flipped, arg, 7 - arg) for arg in pos]
    if len(flipped_pos) == 1:
        return flipped_pos[0]
    return tuple(flipped_pos)

def load_image(path: str, size: tuple[int, int] = None):
    """
    Loads an image from the specified file path and optionally resizes it.

    This function uses Pygame to load an image from the given file path. If a size is provided,
    the image is scaled to the specified dimensions; otherwise, the original image size is retained.

    Parameters:
        path (str): The file path to the image to be loaded.
        size (tuple[int, int], optional): A tuple specifying the desired width and height of the image.
                                          If None, the image is not resized.

    Returns:
        pygame.Surface: The loaded (and optionally resized) image as a Pygame Surface object.
    """
    image = pygame.image.load(path)
    return pygame.transform.scale(image, size) if size else image

def resize_image(image, size):
    """
    Resizes the given image to the specified dimensions.

    This function uses Pygame's transform module to scale an image to the desired size.

    Parameters:
        image (pygame.Surface): The image to be resized. It must be a Pygame Surface object.
        size (tuple): A tuple specifying the new width and height of the image (width, height).

    Returns:
        pygame.Surface: A new Pygame Surface object with the resized image.
    """
    return pygame.transform.scale(image, (size))

def generate_piece_images(flipped: int = 1):
    """
    Generates a dictionary of chess piece images loaded from asset files.

    This function loads chess piece images from a specified directory, resizes them to the 
    configured tile size, and optionally flips the images vertically based on the `flipped` 
    parameter and configuration settings. The images are stored in a dictionary with their 
    notations (e.g., "wK" for white king, "bQ" for black queen) as keys.

    Parameters:
        flipped (int): Determines the flipping behavior of the images. 
                       If 1, flips black pieces if `config.flipped_assets` is True.
                       If -1, flips white pieces if `config.flipped_assets` is True.
                       Defaults to 1.

    Returns:
        dict: A dictionary where keys are piece notations (e.g., "wK", "bQ") and values 
              are the corresponding loaded and processed pygame.Surface images.
    """
    images = dict()
    for file in os.listdir(os.path.join('assets', 'piece', config.piece_asset)):
        filepath = os.path.join('assets', 'piece', config.piece_asset, file)
        notation = os.path.splitext(file)[0]
        image = load_image(filepath, (config.tile_size, config.tile_size))
        if config.flipped_assets and ((flipped == 1 and notation.startswith("b")) or (flipped == -1 and notation.startswith("w"))):
            image = pygame.transform.flip(image, False, True)
        images[notation] = image
    return images

def generate_board_image():
    """
    Generates and returns a resized image of the chessboard based on the specified configuration.

    This function searches for a chessboard image file in the 'assets/board' directory with the 
    name specified by `config.board_asset` and extensions '.jpg' or '.png'. If a matching file 
    is found, it is loaded and resized to fit an 8x8 chessboard grid based on the tile size 
    defined in `config.tile_size`. If no matching file is found, a FileNotFoundError is raised.

    Returns:
        pygame.Surface: A resized image of the chessboard.

    Raises:
        FileNotFoundError: If no board image is found with the specified name and extensions.
    """
    for ext in ['jpg', 'png']:
        filepath = os.path.join('assets', 'board', f"{config.board_asset}.{ext}")
        if os.path.exists(filepath):
            return load_image(filepath, (config.tile_size * 8, config.tile_size * 8))
    raise FileNotFoundError(f"No board image found for {config.board_asset} with extensions .jpg or .png")

def generate_sounds():
    """
    Generates a dictionary of sound objects by loading sound files from the specified
    asset directory and adding custom sounds.

    Utility:
    This function is used to load and organize sound assets for a chess game. It scans
    a directory for sound files, loads them into memory, and maps them to their respective
    names. Additionally, it includes predefined custom sounds such as 'illegal', 'notify',
    and 'tenseconds'.

    Returns:
        dict: A dictionary where the keys are sound names (str) and the values are 
              pygame.mixer.Sound objects representing the loaded sounds.
    """
    sounds = dict()
    for sound in os.listdir(os.path.join('assets', 'sound', config.sound_asset)):
        filepath = os.path.join('assets', 'sound', config.sound_asset, sound)
        name = os.path.splitext(sound)[0]
        sounds[name] = load_sound(filepath)
    custom_sounds = ['illegal', 'notify', 'tenseconds']
    sounds.update({
        name: pygame.mixer.Sound(os.path.join("assets", "sound", f"{name}.ogg"))
        for name in custom_sounds
    })
    return sounds

def play_sound(sounds: dict[str, pygame.Sound], type: str):
    """
    Plays a specific sound from a dictionary of sounds.

    This function retrieves a sound from the provided dictionary using the given type
    and plays it. If the specified type is not found in the dictionary, a ValueError
    is raised.

    Parameters:
        sounds (dict[str, pygame.Sound]): A dictionary where keys are sound types (strings)
                                          and values are pygame.Sound objects.
        type (str): The type of sound to play. This should match one of the keys in the
                    sounds dictionary.

    Raises:
        ValueError: If the specified sound type is not found in the sounds dictionary.
    """
    # Check if the sound type exists in the sounds dictionary
    if type not in sounds:
        raise ValueError(f"Sound type '{type}' not found in the sound library.")
    sounds[type].play()

def debug_print(*args):
    """
    Prints debug information to the console if debugging is enabled.

    This function checks the `debug` attribute in the `config` object. If debugging
    is enabled, it prints the provided arguments to the console. Otherwise, it does nothing.

    Parameters:
        *args: Any
            Variable-length argument list to be printed if debugging is enabled.
    """
    if config.debug:
        print(*args)
        
def singleton(cls):
    """
    A decorator to implement the Singleton design pattern for a class. This ensures that only one 
    instance of the class exists throughout the program's lifecycle. If an instance of the class 
    already exists, it returns the existing instance instead of creating a new one.
    Parameters:
        cls (type): The class to be decorated as a singleton.
    Returns:
        function: A wrapper function that manages the instantiation of the class, ensuring only 
        one instance exists.
    """
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance
