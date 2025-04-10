import cv2
import pygame

from src.constants import Colors


def create_rect_surface(color: tuple[int, int, int], width: int, height: int, border_radius: int,
                        alpha: int = 255, border_width=0, border_color=None) -> pygame.Surface:
    """
    Creates a rectangular Pygame surface with optional border and transparency.

    Args:
        color (tuple[int, int, int]): RGB color of the rectangle.
        width (int): Width of the rectangle.
        height (int): Height of the rectangle.
        border_radius (int): Radius for rounded corners.
        alpha (int, optional): Transparency value (0-255). Default is 255 (opaque).
        border_width (int, optional): Width of the border. Default is 0 (no border).
        border_color (tuple[int, int, int], optional): RGB color of the border.

    Returns:
        pygame.Surface: The created surface with the specified properties.
    """
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(surface, color, (0, 0, width, height), border_radius=border_radius)
    
    if border_color:
        pygame.draw.rect(surface, border_color, (0, 0, width, height), border_width, border_radius=border_radius)
    
    surface.set_alpha(alpha)
    return surface

class Label:
    """
    A class to represent a text label with an optional background.
    """
    def __init__(self, center: tuple[int, int], text: str, font_name: str, font_size: int,
                 color: tuple[int, int, int], background: pygame.Surface = None, 
                 background_pos: tuple[int, int] = None):
        """
        Initializes a Label instance.
        
        Args:
            center (tuple[int, int]): The (x, y) coordinates of the label's center.
            text (str): The text to be displayed.
            font_name (str): The filename of the font (inside 'assets/font/').
            font_size (int): The size of the font.
            color (tuple[int, int, int]): The RGB color of the text.
            background (pygame.Surface, optional): Background image for the label.
            background_pos (tuple[int, int], optional): Position of the background image.
        """
        self.text = text
        self.center = center
        self.font = pygame.font.Font(f"assets/font/{font_name}", font_size)
        self.color = color
        self.background = background
        self.surface = self._create_surface()
        self.rect = self.surface.get_rect(center=center)
        
        if background and background_pos:
            self.background_pos = (background_pos[0] - background.get_width() // 2,
                                   background_pos[1] - background.get_height() // 2)
        else:
            self.background_pos = None

    def draw(self, screen: pygame.Surface):
        """
        Draws the label onto the given screen.
        
        Args:
            screen (pygame.Surface): The surface to draw on.
        """
        if self.background:
            screen.blit(self.background, self.background_pos)
        screen.blit(self.surface, self.rect)

    def update_text(self, new_text: str):
        """
        Updates the text of the label and re-renders it.
        
        Args:
            new_text (str): The new text to display.
        """
        self.text = new_text
        self.surface = self._create_surface()
        self.rect = self.surface.get_rect(center=self.center)

    def _create_surface(self) -> pygame.Surface:
        """
        Creates and returns a rendered text surface.
        
        Returns:
            pygame.Surface: The rendered text surface.
        """
        return self.font.render(self.text, True, self.color)

class RectButton:
    """
    A class representing a rectangular button with optional text and hover effects.
    """
    def __init__(self, x: int, y: int, width: int, height: int, color: tuple[int, int, int],
                 hovered_color: tuple[int, int, int] = None, border_radius: int = 0,
                 text: str = None, font_name: str = None, font_size: int = None,
                 text_color: tuple[int, int, int] = None, command: callable = lambda: None,
                 image: pygame.Surface = None, border_color: tuple[int, int, int] = None):
        """
        Initializes a RectButton instance.

        Args:
            x (int): X-coordinate of the button's center.
            y (int): Y-coordinate of the button's center.
            width (int): Width of the button.
            height (int): Height of the button.
            color (tuple[int, int, int]): RGB color of the button.
            hovered_color (tuple[int, int, int], optional): RGB color when hovered.
            border_radius (int, optional): Radius for rounded corners (default: 0).
            text (str, optional): Text displayed on the button.
            font_name (str, optional): Font name for the text.
            font_size (int, optional): Font size of the text.
            text_color (tuple[int, int, int], optional): RGB color of the text.
            command (callable, optional): Function to execute on click (default: no action).
            image (pygame.Surface, optional): Optional image displayed on the button.
            border_color (tuple[int, int, int], optional): Optional border color.
        """
        self.width, self.height = width, height
        self.x, self.y = x - width // 2, y - height // 2
        self.rect = pygame.Rect(self.x, self.y, width, height)
        self.color = color
        self.hovered_color = hovered_color
        self.border_radius = border_radius
        self.border_color = border_color
        self.command = command
        self.is_hovered = False
        self.image = image
        
        self.label = Label(self.rect.center, text, font_name, font_size, text_color) if text else None
        
        if image:
            self.image_pos = (
                self.rect.centerx - image.get_width() // 2,
                self.rect.centery - image.get_height() // 2
            )

    def draw(self, screen: pygame.Surface):
        """
        Draws the button on the given screen

        Args:
            screen (pygame.Surface): The surface to draw on.
        
        """
        pygame.draw.rect(
            screen, self.hovered_color if self.is_hovered and self.hovered_color else self.color,
            self.rect, border_radius=self.border_radius
        )
        
        if self.label:
            self.label.draw(screen)
        
        if self.border_color:
            pygame.draw.rect(screen, self.border_color, self.rect, int(self.height // 11), self.border_radius)
        
        if self.image:
            screen.blit(self.image, self.image_pos)

    def update(self, mouse_pos: tuple[int, int]):
        """
        Updates the hover state based on the mouse position.

        Args:
            mouse_pos (tuple): The position of the mouse
        """
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def handle_click(self):
        """
        Executes the assigned command if the button is clicked.
        """
        if self.is_clicked():
            self.command()

    def is_clicked(self) -> bool:
        """
        Checks if the button is currently clicked.
        """
        return self.rect.collidepoint(pygame.mouse.get_pos())

    def update_text(self, new_text: str):
        """
        Updates the button's text.
        
        Args:
            new_text (str): The new text to display.
        """
        if self.label:
            self.label.update_text(new_text)

    def update_color(self, color: tuple[int, int, int], hovered_color: tuple[int, int, int]):
        """
        Updates the button's color and hover color.
        
        Args:
            color (tuple[int, int, int]): new RGB color of the button.
            hovered_color (tuple[int, int, int]): new RGB color when hovered
        """
        self.color = color
        self.hovered_color = hovered_color

class RadioButton:
    """
    A class representing a circular radio button with selectable states.
    """
    def __init__(self, x: int, y: int, radius: int, width: int, color: tuple[int, int, int], 
                 state: bool, command: callable = lambda: None):
        """
        Initializes a RadioButton instance.

        Args:
            x (int): X-coordinate of the button's center.
            y (int): Y-coordinate of the button's center.
            radius (int): Radius of the button.
            width (int): Thickness of the outer circle.
            color (tuple[int, int, int]): RGB color of the button.
            state (bool): Initial selection state of the button.
            command (callable, optional): Function to execute when clicked (default: no action).
        """
        self.x, self.y = x, y
        self.radius = radius
        self.rect = pygame.Rect(self.x - radius, self.y - radius, radius * 2, radius * 2)
        self.width = width
        self.color = color
        self.state = state
        self.command = command

    def draw(self, screen: pygame.Surface):
        """
        Draws the radio button on the given screen.

        Args:
            screen (pygame.Surface): The surface to draw on.
        """
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius, self.width)
        if self.state:
            pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius // 2)
        
    def update(self, mouse_pos: tuple[int, int]):
        """
        Placeholder method for updating the button state.

        Args:
            mouse_pos (tuple[int, int]): The position of the mouse.
        """
        pass
    
    def handle_click(self):
        """
        Placeholder method for handling click events.
        """
        pass
    
    def is_clicked(self) -> bool:
        """
        Checks if the radio button is currently clicked.
        """
        return self.rect.collidepoint(pygame.mouse.get_pos())

class VideoPlayer:
    """
    A class for playing videos using OpenCV and displaying frames in Pygame.
    """
    def __init__(self, video_path: str, width: int, height: int):
        """
        Initializes a VideoPlayer instance.

        Args:
            video_path (str): Path to the video file.
            width (int): Desired video width.
            height (int): Desired video height (maintains aspect ratio).
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        _, frame = self.cap.read()
        self.frame_width = width
        self.frame_height = int(width * (frame.shape[0] / frame.shape[1]))
    
    def play(self, screen: pygame.Surface):
        """
        Plays the video by rendering frames onto the given Pygame screen.
        
        Args:
            screen (pygame.Surface): The Pygame surface where the video will be displayed.
        """
        ret, frame = self.cap.read()
        
        if not ret:
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            ret, frame = self.cap.read()
        
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        
        screen.blit(frame, (0, 0))
