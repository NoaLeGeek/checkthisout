import random

import pygame

from src.config import config
from src.utils import load_image
from src.scenes.game import Game
from src.scenes.scene import Scene
from src.board.player import Player
from src.constants import Fonts, Colors, available_rule
from src.gui import RectButton, Label, VideoPlayer, create_rect_surface, RadioButton
from src.ia.negamax import NegamaxAI, RandomAI
from src.ia.ml.loader import load_model, load_model_from_checkpoint

def str_to_ia(ia: str, color: int, depth=None):
    """
    Converts a string representation of an AI type to the corresponding AI object.
    
    Args:
        ia (str): A string representing the AI type ('random', 'negamax', or 'neural_network').
        color (int): The color the AI will play as (1 for white, -1 for black).
        depth (int, optional): The depth for the 'negamax' AI. Defaults to None.

    Returns:
        object: The corresponding AI object (RandomAI, NegamaxAI, or a Neural Network model).
    """
    if 'random' in ia:
        ia = 'random'
    elif 'negamax' in ia:
        ia = 'negamax'
    elif 'neural_network' in ia:
        ia = 'neural_network'
    return {'random': RandomAI(color), 
            'negamax': NegamaxAI(color, depth), 
            'neural_network': load_model('data/models/v1', color)}[ia]

class SetupMenu(Scene):
    """
    Represents the setup menu scene where the player can choose between different game modes
    and configure various game rules.
    """

    def __init__(self):
        """
        Initializes the setup menu scene with the player objects and background image.
        """
        super().__init__()
        self.player1 = Player(1)
        self.player2 = Player(-1)

        # Frame area where buttons and options are displayed
        self.frame = pygame.Rect(config.width*0.2, config.height*0.2, config.width*0.6, config.height*0.6)
        self.bg = load_image('assets/images/setup_bg.jpg', (config.width, config.height))

    def create_buttons(self):
        """
        Creates the buttons for different game modes: Player vs Player, Player vs AI, and AI vs AI.
        Also creates the back button to navigate to the previous menu.
        """
        button_width = config.width * 0.27
        button_height = config.height * 0.1
        font_size = int(button_height * 0.7)

        self.buttons = {
            "player vs player": RectButton(
                x=config.width*0.6, 
                y=config.height*0.3, 
                width=button_width, 
                height=button_height, 
                border_radius=int(button_height//2), 
                color=Colors.LIGHT_GRAY.value, 
                hovered_color=Colors.WHITE.value,
                text='PLAYER VS PLAYER',
                font_name=Fonts.GEIZER, 
                font_size=font_size, 
                text_color=Colors.BLACK.value, 
                command=lambda: self.manager.go_to(Game(self.player1, self.player2))
            ),
            "player vs ia": RectButton(
                x=config.width*0.6, 
                y=config.height*0.5, 
                width=button_width, 
                height=button_height, 
                border_radius=int(button_height//2), 
                color=Colors.LIGHT_GRAY.value, 
                hovered_color=Colors.WHITE.value, 
                text='PLAYER VS IA', 
                font_name=Fonts.GEIZER, 
                font_size=font_size, 
                text_color=Colors.BLACK.value, 
                command=lambda: self.manager.go_to(PlayerVsIaMenu())
            ),
            "ia vs ia": RectButton(
                x=config.width*0.6, 
                y=config.height*0.7, 
                width=button_width, 
                height=button_height, 
                border_radius=int(button_height//2), 
                color=Colors.LIGHT_GRAY.value, 
                hovered_color=Colors.WHITE.value,
                text='IA VS IA', 
                font_name=Fonts.GEIZER, 
                font_size=font_size, 
                text_color=Colors.BLACK.value, 
                command=lambda: self.manager.go_to(IaVsIaMenu())
            ),
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

        self.rule_buttons = {
            rule: RadioButton(
                x=config.width*0.415,
                y=config.height*0.305+i*button_height,
                radius=config.height*0.03,
                width=int(config.height*0.005),
                color=Colors.WHITE.value, 
                state=config.rules[rule],
            ) for i, rule in enumerate(available_rule)
        }

    def create_labels(self):
        """
        Creates the labels for different game rules and displays them.
        """
        self.labels = {
            rule: Label(
                center=(config.width*0.3, config.height*0.305+(i*config.height*0.1)),
                text=rule.replace('_', ' ') if rule != 'king_of_the_hill' else 'koth',
                font_name=Fonts.GEIZER,
                font_size=int(config.height*0.08),
                color=Colors.WHITE.value
            )
            for i, rule in enumerate(available_rule)
        }

    def render(self, screen):
        """
        Renders the setup menu scene, including the background, frame, and buttons.
        
        Args:
            screen (pygame.Surface): The pygame screen object where the scene will be rendered.
        """
        screen.blit(self.bg, (0, 0))
        pygame.draw.rect(screen, Colors.DARK_GRAY.value, self.frame, border_radius=int(config.height*0.08))
        pygame.draw.rect(screen, Colors.WHITE.value, self.frame, width=1, border_radius=int(config.height*0.08))
        super().render(screen)
        for button in self.rule_buttons.values():
            button.draw(screen)
    
    def handle_event(self, event):
        """
        Handles events like mouse clicks and updates the game rules accordingly.

        Args:
            event (pygame.event): The event to be handled.
        """
        super().handle_event(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                for rule, button in self.rule_buttons.items():
                    if button.is_clicked():
                        self.update_rule(rule, button)

    def update_rule(self, rule, button):
        """
        Updates the state of a specific game rule when clicked.

        Args:
            rule (str): The rule being toggled.
            button (RadioButton): The button representing the rule.
        """
        for r in available_rule:
            if config.rules[r]:
                config.rules[r] = False
        config.rules[rule] = not config.rules[rule]
        button.state = True
        for b in self.rule_buttons.values():
            if button is not b:
                b.state = False


class PlayerVsIaMenu(Scene):
    """
    Represents the menu for a player vs AI game mode.
    Handles the selection of player color, AI type, and game depth.
    """

    def __init__(self):
        """
        Initializes the menu with a human player (player1) and a randomly-selected AI (player2).
        Sets up the graphical elements like frame, background, and initial values.
        """
        self.player1 = Player(1)  # Human player with color 1 (white)
        self.player2 = RandomAI(-1)  # AI player with color -1 (black)
        self.selected_color = 1  # Default selected color is white
        self.frame = pygame.Rect(config.width*0.2, config.height*0.2, config.width*0.6, config.height*0.6)  # Frame for UI
        self.depth = 2  # Default AI depth
        self.bg = load_image('assets/images/ia_bg.jpeg', (config.width, config.height))  # Background image
        super().__init__()

    def create_buttons(self) -> None:
        """
        Creates the UI buttons for starting the game, choosing colors, adjusting depth, and selecting AI type.
        """
        self.buttons = {
            "play": RectButton(  # Play button
                x=config.width*0.65, 
                y=config.height*0.68, 
                width=config.width*0.2, 
                height=config.height*0.1, 
                border_radius=int(config.height*0.715/2),
                color=Colors.LIGHT_GRAY.value, 
                hovered_color=Colors.WHITE.value,
                text='check this out !',
                font_name=Fonts.GEIZER, 
                font_size=int(config.height*0.06), 
                text_color=Colors.BLACK.value, 
                command=self.start_game
            ),
            'white': RectButton(  # White color button
                x=config.width*0.27,
                y=config.height*0.68,
                width=config.tile_size,
                height=config.tile_size,
                border_radius=1,
                color=Colors.WHITE.value, 
                image=load_image('assets/piece/alpha/wK.svg', (config.tile_size, config.tile_size)),
                command=lambda: self.update_color(1)
            ),
            'black': RectButton(  # Black color button
                x=config.width*0.37,
                y=config.height*0.68,
                width=config.tile_size,
                height=config.tile_size,
                color=Colors.BLACK.value, 
                image=load_image('assets/piece/alpha/bK.svg', (config.tile_size, config.tile_size)),
                command=lambda: self.update_color(-1)
            ),
            'random_color': RectButton(  # Random color button
                x=config.width*0.47,
                y=config.height*0.68,
                width=config.tile_size,
                height=config.tile_size,
                color=Colors.GRAY.value, 
                command=lambda: self.update_color('random_color'),
                text='?',
                text_color=Colors.WHITE.value,
                font_size=int(config.tile_size*0.75),
                font_name=Fonts.GEIZER
            ),
            'decrease_depth': RectButton(  # Decrease AI depth button
                x=config.width*0.67,
                y=config.height*0.305 + config.height*0.1, 
                width=config.height*0.1,
                height=config.height*0.1,
                color=Colors.DARK_GRAY.value,
                text='<',
                text_color=Colors.GRAY.value if not config.rules['classic'] and not config.rules['chess960'] else Colors.WHITE.value,
                font_size=int(config.height*0.1),
                font_name=Fonts.GEIZER,
                command=lambda: self.update_depth(-1) if config.rules['classic'] or config.rules['chess960'] else lambda: None
             ),
            'increase_depth': RectButton(  # Increase AI depth button
                x=config.width*0.745,
                y=config.height*0.305 + config.height*0.1, 
                width=config.height*0.1,
                height=config.height*0.1,
                color=Colors.DARK_GRAY.value,
                text='>',
                text_color=Colors.GRAY.value if not config.rules['classic'] and not config.rules['chess960'] else Colors.WHITE.value,
                font_size=int(config.height*0.1),
                font_name=Fonts.GEIZER,
                command=lambda: self.update_depth(1) if config.rules['classic'] or config.rules['chess960'] else lambda: None
            ),
            'back': RectButton(  # Back button
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

        self.ia_buttons = {
            ia : RadioButton(  # AI selection buttons (random, negamax, neural network)
                x=config.width*0.52,
                y=config.height*0.305 + i*config.height * 0.1,
                radius=config.height*0.03,
                width=int(config.height*0.005),
                color=Colors.GRAY.value if (ia in ['neural_network', 'negamax'] and not config.rules['classic'] and not config.rules['chess960']) else Colors.WHITE.value,
                state=False if ia != 'random_ia' else True
            )
            for i, ia in enumerate(['random_ia', 'negamax', 'neural_network'])
        }
        self.buttons.update(self.ia_buttons)

    def create_labels(self) -> None:
        """
        Creates the labels for the game (e.g., depth label, AI type label).
        """
        self.labels = {
            'depth': Label(  # Label for depth selection
                center=(config.width*0.6, config.height*0.305 + config.height*0.1),
                text='depth',
                font_name=Fonts.GEIZER,
                font_size=int(config.height*0.07),
                color=Colors.GRAY.value if (not config.rules['classic'] and not config.rules['chess960']) else Colors.WHITE.value,
            ),
            'depth_num': Label(  # Label to display the current depth
                center=(config.width*0.7075, config.height*0.305 + config.height*0.1),
                text=str(self.depth),
                font_name=Fonts.GEIZER,
                font_size=int(config.height*0.07),
                color=Colors.GRAY.value if (not config.rules['classic'] and not config.rules['chess960']) else Colors.WHITE.value,
            )
        }
        self.ia_labels = {
            ia : Label(  # Labels for AI types (random, negamax, neural network)
                center=(config.width*0.35, config.height*0.305 + (i*config.height*0.1)),
                text=ia,
                font_name=Fonts.GEIZER,
                font_size=int(config.height*0.08),
                color=Colors.GRAY.value if (ia in ['neural network', 'negamax'] and not config.rules['classic'] and not config.rules['chess960']) else Colors.WHITE.value,
            )
            for i, ia in enumerate(['random', 'negamax', 'neural network'])
        }
        self.labels.update(self.ia_labels)

    def update_color(self, color: int | str) -> None:
        """
        Updates the selected player color and AI color accordingly.
        
        Args:
            color (int|str): The selected color. Can be an integer (1 for white, -1 for black) or a string ('random_color').
        """
        if color != 'random_color':
            self.player1.color = color
            self.player2.color = -color       
        self.selected_color = color
    
    def render(self, screen: pygame.Surface) -> None:
        """
        Renders the UI elements and displays the background, frame, and buttons.
        Highlights the selected color button with a green border.
        
        Args: 
            screen: The pygame surface on which to render the UI elements.
        """
        screen.blit(self.bg, (0, 0))
        pygame.draw.rect(screen, Colors.DARK_GRAY.value, self.frame, border_radius=int(config.height*0.08))
        pygame.draw.rect(screen, Colors.WHITE.value, self.frame, width=1, border_radius=int(config.height*0.08))
        super().render(screen)
        if self.selected_color == 1:
            pygame.draw.rect(screen, Colors.GREEN.value, self.buttons['white'].rect, 2)
        elif self.selected_color == -1:
            pygame.draw.rect(screen, Colors.GREEN.value, self.buttons['black'].rect, 2)
        elif self.selected_color == 'random_color':
            pygame.draw.rect(screen, Colors.GREEN.value, self.buttons['random_color'].rect, 2)

    def handle_event(self, event: pygame.event.Event) -> None:
        """
        Handles user interactions such as button clicks for AI and depth changes.
        
        Args:
            event (pygame.event.Event): The event to handle, typically a mouse click event.
        """
        super().handle_event(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                for ia, button in self.ia_buttons.items():
                    if button.is_clicked() and not (ia in ['neural_network', 'negamax'] and not config.rules['classic'] and not config.rules['chess960']):
                        self.update_ia(ia, button)

    def update_ia(self, ia: str, button: RadioButton) -> None:
        """
        Updates the AI selection based on the button clicked.
        
        Args: 
            ia (str): The AI type selected ('random_ia', 'negamax', or 'neural_network').
            button (RadioButton): The button that was clicked.
        """
        button.state = True
        self.player2 = str_to_ia(ia, self.player2.color, self.depth)
        for b in self.ia_buttons.values():
            if button is not b:
                b.state = False

    def update_depth(self, n: int) -> None:
        """
        Adjusts the depth of the AI based on the user’s input.
        The depth is constrained to be between 1 and 5.
        
        Args: 
            n (int): The amount to increase/decrease the depth (positive or negative integer).
        """
        self.depth = (self.depth + n)
        if self.depth > 5:
            self.depth = 1
        elif self.depth < 1:
            self.depth = 5
        self.labels['depth_num'].update_text(str(self.depth))

    def start_game(self) -> None:
        """
        Starts the game based on the selected settings (color, AI, depth).
        """
        if self.selected_color == 'random_color':
            self.player1.color = random.choice((1, -1))
            self.player2.color = -self.player1.color
        self.manager.go_to(Game(self.player1 if self.player1.color == 1 else self.player2, self.player1 if self.player1.color == -1 else self.player2))



class IaVsIaMenu(Scene):
    """
    Represents the IA vs IA menu scene where the user can configure the depth of both AIs,
    choose the type of AI for each player, and start the game.
    """

    def __init__(self):
        """
        Initializes the IA vs IA menu scene with the player objects (AI), background image, and frame.
        """
        self.player1 = RandomAI(1)  # Player 1 is an AI (RandomAI)
        self.player2 = RandomAI(-1)  # Player 2 is an AI (RandomAI)
        
        # Frame area where buttons and options are displayed
        self.frame = pygame.Rect(config.width * 0.15, config.height * 0.18, config.width * 0.7, config.height * 0.64)
        
        # Kings' images for white and black
        self.white_king = load_image('assets/piece/alpha/wK.svg', (config.tile_size, config.tile_size))
        self.white_rect = pygame.Rect(config.width * 0.45 - config.tile_size // 2, config.height * 0.2, config.tile_size, config.tile_size)
        self.black_king = load_image('assets/piece/alpha/bK.svg', (config.tile_size, config.tile_size))
        self.black_rect = pygame.Rect(config.width * 0.66 - config.tile_size // 2, config.height * 0.2, config.tile_size, config.tile_size)
        
        # Depth levels for AI players
        self.depth = {1: 2, 2: 2}
        
        # Background image
        self.bg = load_image('assets/images/ia_bg.jpeg', (config.width, config.height))
        
        super().__init__()

    def create_buttons(self):
        """
        Creates the buttons for selecting IA types for both players, adjusting the depth of AIs, and starting the game.
        """
        self.buttons = {
            'back': RectButton(
                x=config.width * 0.955,
                y=config.height * 0.08, 
                width=config.height * 0.1,
                height=config.height * 0.1,
                color=Colors.LIGHT_GRAY.value,
                hovered_color=Colors.WHITE.value,
                text='<-',
                text_color=Colors.DARK_GRAY.value,
                font_size=int(config.height * 0.1),
                font_name=Fonts.GEIZER,
                command=self.manager.go_back
            ),
            'decrease_depth1': RectButton(
                x=config.width * 0.56,
                y=config.height * 0.385 + config.height * 0.1, 
                width=config.height * 0.07,
                height=config.height * 0.07,
                color=Colors.DARK_GRAY.value,
                text='<',
                text_color=Colors.GRAY.value if not config.rules['classic'] and not config.rules['chess960'] else Colors.WHITE.value,
                font_size=int(config.height * 0.07),
                font_name=Fonts.GEIZER,
                command=lambda: self.update_depth(-1, 1) if config.rules['classic'] or config.rules['chess960'] else lambda: None
            ),
            'increase_depth1': RectButton(
                x=config.width * 0.62,
                y=config.height * 0.385 + config.height * 0.1, 
                width=config.height * 0.07,
                height=config.height * 0.07,
                color=Colors.DARK_GRAY.value,
                text='>',
                text_color=Colors.GRAY.value if not config.rules['classic'] and not config.rules['chess960'] else Colors.WHITE.value,
                font_size=int(config.height * 0.07),
                font_name=Fonts.GEIZER,
                command=lambda: self.update_depth(1, 1) if config.rules['classic'] or config.rules['chess960'] else lambda: None
            ),
            'decrease_depth2': RectButton(
                x=config.width * 0.77,
                y=config.height * 0.385 + config.height * 0.1, 
                width=config.height * 0.07,
                height=config.height * 0.07,
                color=Colors.DARK_GRAY.value,
                text='<',
                text_color=Colors.GRAY.value if not config.rules['classic'] and not config.rules['chess960'] else Colors.WHITE.value,
                font_size=int(config.height * 0.07),
                font_name=Fonts.GEIZER,
                command=lambda: self.update_depth(-1, 2) if config.rules['classic'] or config.rules['chess960'] else lambda: None
            ),
            'increase_depth2': RectButton(
                x=config.width * 0.83,
                y=config.height * 0.385 + config.height * 0.1, 
                width=config.height * 0.07,
                height=config.height * 0.07,
                color=Colors.DARK_GRAY.value,
                text='>',
                text_color=Colors.GRAY.value if not config.rules['classic'] and not config.rules['chess960'] else Colors.WHITE.value,
                font_size=int(config.height * 0.07),
                font_name=Fonts.GEIZER,
                command=lambda: self.update_depth(1, 2) if config.rules['classic'] or config.rules['chess960'] else lambda: None
            ),
            "play": RectButton(
                x=config.width * 0.5, 
                y=config.height * 0.715, 
                width=config.width * 0.2, 
                height=config.height * 0.1, 
                border_radius=int(config.height * 0.715 / 2),
                color=Colors.LIGHT_GRAY.value, 
                hovered_color=Colors.WHITE.value,
                text='check this out !',
                font_name=Fonts.GEIZER, 
                font_size=int(config.height * 0.06), 
                text_color=Colors.BLACK.value, 
                command=lambda: self.manager.go_to(Game(self.player1, self.player2))
            )
        }
        
        # AI selection buttons for white and black players
        self.white_buttons = {
            ia + '1': RadioButton(
                x=config.width * 0.45,
                y=config.height * 0.38 + i * config.height * 0.1,
                radius=config.height * 0.03,
                width=int(config.height * 0.005),
                color=Colors.GRAY.value if (ia in ['neural_network', 'negamax'] and not config.rules['classic'] and not config.rules['chess960']) else Colors.WHITE.value,
                state=False if ia != 'random_ia' else True
            )
            for i, ia in enumerate(['random_ia', 'negamax', 'neural_network'])
        }
        
        self.black_buttons = {
            ia + '2': RadioButton(
                x=config.width * 0.66,
                y=config.height * 0.38 + i * config.height * 0.1,
                radius=config.height * 0.03,
                width=int(config.height * 0.005),
                color=Colors.GRAY.value if (ia in ['neural_network', 'negamax'] and not config.rules['classic'] and not config.rules['chess960']) else Colors.WHITE.value,
                state=False if ia != 'random_ia' else True
            )
            for i, ia in enumerate(['random_ia', 'negamax', 'neural_network'])
        }
        
        self.buttons.update(self.white_buttons)
        self.buttons.update(self.black_buttons)

    def create_labels(self):
        """
        Creates the labels for the IA settings, including depth and IA types for both players.
        """
        self.labels = {
            'depth': Label(
                center=(config.width * 0.615, config.height * 0.385 + config.height * 0.1),
                text='depth                      depth',
                font_name=Fonts.GEIZER,
                font_size=int(config.height * 0.05),
                color=Colors.GRAY.value if (not config.rules['classic'] and not config.rules['chess960']) else Colors.WHITE.value,
            ),
            'depth1_num': Label(
                center=(config.width * 0.59, config.height * 0.385 + config.height * 0.1),
                text=str(self.depth[1]),
                font_name=Fonts.GEIZER,
                font_size=int(config.height * 0.05),
                color=Colors.GRAY.value if (not config.rules['classic'] and not config.rules['chess960']) else Colors.WHITE.value,
            ),
            'depth2_num': Label(
                center=(config.width * 0.8, config.height * 0.385 + config.height * 0.1),
                text=str(self.depth[2]),
                font_name=Fonts.GEIZER,
                font_size=int(config.height * 0.05),
                color=Colors.GRAY.value if (not config.rules['classic'] and not config.rules['chess960']) else Colors.WHITE.value,
            ),
        }

        # IA labels for both players
        ia_labels = {
            ia: Label(
                center=(config.width * 0.29, config.height * 0.38 + (i * config.height * 0.1)),
                text=ia,
                font_name=Fonts.GEIZER,
                font_size=int(config.height * 0.08),
                color=Colors.GRAY.value if (ia in ['neural network', 'negamax'] and not config.rules['classic'] and not config.rules['chess960']) else Colors.WHITE.value,
            )
            for i, ia in enumerate(['random', 'negamax', 'neural network'])
        }

        self.labels.update(ia_labels)

    def render(self, screen):
        """
        Renders the IA vs IA menu scene, including the background, buttons, and labels.
        
        Args:
            screen (pygame.Surface): The pygame screen object where the scene will be rendered.
        """
        screen.blit(self.bg, (0, 0))
        pygame.draw.rect(screen, Colors.DARK_GRAY.value, self.frame, border_radius=int(config.height * 0.1))
        pygame.draw.rect(screen, Colors.WHITE.value, self.frame, width=1, border_radius=int(config.height * 0.1))
        pygame.draw.rect(screen, Colors.WHITE.value, self.white_rect)
        pygame.draw.rect(screen, Colors.BLACK.value, self.black_rect)
        screen.blit(self.white_king, self.white_rect)
        screen.blit(self.black_king, self.black_rect)
        super().render(screen)

    def update_depth(self, n, num):
        """
        Updates the depth level for a specific AI player (either 1 or 2).
        
        Args:
            n (int): The amount to increase or decrease the depth.
            num (int): The player number (1 or 2) whose depth level is being updated.
        """
        self.depth[num] = (self.depth[num] + n)
        if self.depth[num] > 5:
            self.depth[num] = 1
        elif self.depth[num] < 1:
            self.depth[num] = 5

        self.labels[f'depth{num}_num'].update_text(str(self.depth[num]))

    def update_ia(self, num, ia, button, button_dict):
        """
        Updates the selected AI for a specific player (either 1 or 2).
        
        Args:
            num (int): The player number (1 or 2).
            ia (str): The IA type to be assigned to the player (e.g., 'random_ia').
            button (RadioButton): The button representing the IA selection.
            button_dict (dict): The dictionary containing all the IA buttons.
        """
        button.state = True
        if num == 1:
            self.player1 = str_to_ia(ia, 1, self.depth[1])
        else:
            self.player2 = str_to_ia(ia, -1, self.depth[2])
        for b in button_dict.values():
            if button is not b:
                b.state = False

    def handle_event(self, event):
        """
        Handles events like mouse clicks and updates IA selections or depth levels accordingly.

        Args:
            event (pygame.event): The event to be handled.
        """
        super().handle_event(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                for ia, button in self.white_buttons.items():

                    if button.is_clicked() and not (ia in ['neural_network1', 'negamax1'] and not config.rules['classic'] and not config.rules['chess960']): 
                        self.update_ia(1, ia, button, self.white_buttons)
                for ia, button in self.black_buttons.items():
                    if button.is_clicked() and not (ia in ['neural_network2', 'negamax2'] and not config.rules['classic'] and not config.rules['chess960']): 
                        self.update_ia(2, ia, button, self.black_buttons)
        