import webbrowser

from src.config import config
from src.scenes.game import Game
from src.scenes.scene import Scene
from src.board.player import Player
from src.scenes.setup import SetupMenu
from src.constants import Fonts, Colors
from src.scenes.settings import SettingsMenu
from src.gui import RectButton, Label, VideoPlayer, create_rect_surface


class MainMenu(Scene):
    """
    Represents the main menu of the game, providing navigation to different sections such as
    starting the game, accessing settings, viewing credits, or quitting the application.
    """

    def __init__(self):
        """
        Initializes the main menu with a background video and prepares buttons for navigation.
        """
        super().__init__()
        self.video_player = VideoPlayer("assets/video/chess.mp4", config.width, config.height)

    def create_buttons(self):
        """
        Creates the interactive buttons for the main menu, allowing the user to start the game,
        access settings, view credits, read the rules, or quit the game.
        """
        button_width = config.width * 0.27
        button_height = config.height * 0.1
        font_size_1 = int(button_height * 0.75)
        font_size_2 = int(button_height * 0.58)

        self.buttons = {
            "play": RectButton(
                x=config.width * 0.5,
                y=config.height * 0.46,
                width=button_width,
                height=button_height,
                border_radius=int(button_height // 2),
                color=Colors.LIGHT_GRAY.value, 
                hovered_color=Colors.WHITE.value,
                text='PLAY',
                font_name=Fonts.GEIZER,
                font_size=font_size_1,
                text_color=Colors.DARK_GRAY.value,
                border_color=Colors.DARK_GRAY.value,
                command=lambda: self.manager.go_to(SetupMenu())
            ),
            "settings": RectButton(
                x=config.width * 0.5,
                y=config.height * 0.66,
                width=button_width,
                height=button_height,
                border_radius=int(button_height // 2),
                color=Colors.LIGHT_GRAY.value, 
                hovered_color=Colors.WHITE.value,
                text='SETTINGS',
                font_name=Fonts.GEIZER,
                font_size=font_size_1,
                text_color=Colors.DARK_GRAY.value,
                command=lambda: self.manager.go_to(SettingsMenu()),
                border_color=Colors.DARK_GRAY.value
            ),
            "quit": RectButton(
                x=config.width * 0.5,
                y=config.height * 0.86,
                width=button_width,
                height=button_height,
                border_radius=int(button_height // 2),
                color=Colors.LIGHT_GRAY.value, 
                hovered_color=Colors.WHITE.value,
                text='QUIT',
                font_name=Fonts.GEIZER,
                font_size=font_size_1,
                text_color=Colors.DARK_GRAY.value,
                command=quit,
                border_color=Colors.DARK_GRAY.value
            ),
            "rules": RectButton(
                x=config.width * 0.1,
                y=config.height * 0.94,
                width=config.width * 0.15,
                height=config.height * 0.08,
                border_radius=int(config.height * 0.08 // 2),
                color=Colors.LIGHT_GRAY.value, 
                hovered_color=Colors.WHITE.value,
                text='RULES',
                font_name=Fonts.GEIZER,
                font_size=font_size_2,
                text_color=Colors.DARK_GRAY.value,
                command=lambda: webbrowser.open(
                    "https://lechiquiers.com/blogs/news/comment-jouer-aux-echecs-pour-les-debutants-installation-coups-et-regles-de-base-expliques"
                ),
                border_color=Colors.DARK_GRAY.value
            ),
            "credits": RectButton(
                x=config.width * 0.9,
                y=config.height * 0.94,
                width=config.width * 0.15,
                height=config.height * 0.08,
                border_radius=int(config.height * 0.08 // 2),
                color=Colors.LIGHT_GRAY.value, 
                hovered_color=Colors.WHITE.value,
                text='CREDITS',
                font_name=Fonts.GEIZER,
                font_size=font_size_2,
                text_color=Colors.DARK_GRAY.value,
                command=lambda: self.manager.go_to(CreditsMenu()),
                border_color=Colors.DARK_GRAY.value
            )
        }

    def create_labels(self):
        """
        Creates the labels for the main menu, including the game title.
        """
        title_background = create_rect_surface(
            Colors.WHITE.value, config.width*0.6, config.height*0.22,
            border_radius=int(config.height*0.055),
            border_width=int(config.height*0.01),
            border_color=Colors.DARK_GRAY.value
        )
        self.labels = {
            "title": Label(
                (config.width*0.5, config.height*0.185), 'CheckThisOut',
                Fonts.ONE_SLICE, int(config.height*0.2), Colors.DARK_GRAY.value,
                title_background, (config.width*0.5, config.height*0.17)
            )
        }
        
    def render(self, screen):
        """
        Renders the main menu scene, including the background video, labels, and buttons.
        
        Args:
            screen (pygame.Surface): The pygame screen object where the scene will be rendered.
        """
        self.video_player.play(screen)
        super().render(screen)



class CreditsMenu(Scene):
    """
    Represents the credits menu, displaying the game's credits and allowing users to return to the previous menu.
    """

    def __init__(self):
        """
        Initializes the credits menu without additional components.
        """
        super().__init__()

    def create_buttons(self):
        """
        Creates the back button to return to the previous menu.
        """
        self.buttons = {
            'back': RectButton(
                x=config.width*0.955,
                y=config.height*0.08, 
                width=config.height*0.1,
                height=config.height*0.1,
                color=Colors.LIGHT_GRAY.value,
                hovered_color=Colors.WHITE.value,
                text='<-',
                text_color=Colors.DARK_GRAY.value,
                font_size=int(config.height*0.1),
                font_name=Fonts.GEIZER,
                command=self.manager.go_back
            )
        }

    def create_labels(self):
        """
        Loads and creates the label displaying the credits from a text file.
        """
        with open('assets/text/credits.txt', 'r', encoding='utf-8') as file:
            text = file.read()
        self.labels = {
            "credits": Label(
                center=(config.width*0.5, config.height*0.5), 
                text=text,
                font_name=Fonts.GEIZER, 
                font_size=int(config.height*0.1), 
                color=Colors.WHITE.value
            )
        }
