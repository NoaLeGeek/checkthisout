import pygame

from src.config import config
from src.scenes.scene import Scene
from src.gui import Label, RectButton, create_rect_surface
from src.constants import Colors, Fonts, available_board, available_piece, available_sound
from src.utils import generate_piece_images, generate_board_image, load_image, generate_sounds, resize_image


class SettingsMenu(Scene):
    """
    Represents the settings menu where the user can configure various game settings such as 
    board assets, piece assets, sound settings, and volume controls.
    """

    def __init__(self):
        """
        Initializes the settings menu with volume control, assets menus, and background image.
        """
        super().__init__()
        self.volume_bar = VolumeBar()
        self.piece_assets_menu = PieceAssetsMenu(self)
        self.board_assets_menu = BoardAssetsMenu(self)
        self.sound_assets_menu = SoundAssetsMenu(self)
        self.assets_menu = None
        self.filter = create_rect_surface(Colors.BLACK.value, config.width, config.height, 0, 210)
        self.frame = pygame.Rect(config.width*0.3, config.height*0.2, config.width*0.4, config.height*0.6)
        self.volume_icon = load_image('assets/images/volume.png', (config.height*0.1, config.height*0.1))
        self.volume_icon_rect = self.volume_icon.get_rect(center=(config.width*0.353, config.height*0.732))
        self.bg = load_image('assets/images/settings_bg.jpeg', (config.width, config.height))

    def create_buttons(self):
        """
        Creates the buttons for piece assets, board assets, sound assets, and back to the previous menu.
        """
        button_width = config.width * 0.3
        button_height = config.height * 0.1
        button_x = config.width * 0.5
        font_size = int(button_height * 0.5)

        self.buttons = {
            "piece": RectButton(
                x=button_x,
                y=config.height * 0.3,
                width=button_width,
                height=button_height,
                color=Colors.LIGHT_GRAY.value, 
                hovered_color=Colors.WHITE.value,
                text='PIECE ASSETS',
                font_name=Fonts.GEIZER, 
                font_size=font_size, 
                text_color=Colors.BLACK.value, 
                command=lambda: self.change_menu(self.piece_assets_menu)
            ),
            "board": RectButton(
                x=button_x,
                y=config.height * 0.45,
                width=button_width,
                height=button_height,
                color=Colors.LIGHT_GRAY.value, 
                hovered_color=Colors.WHITE.value,
                text='BOARD ASSETS',
                font_name=Fonts.GEIZER, 
                font_size=font_size, 
                text_color=Colors.BLACK.value, 
                command=lambda: self.change_menu(self.board_assets_menu)
            ),
            "sound": RectButton(
                x=button_x,
                y=config.height * 0.6,
                width=button_width,
                height=button_height,
                color=Colors.LIGHT_GRAY.value, 
                hovered_color=Colors.WHITE.value,
                text='SOUND ASSETS',
                font_name=Fonts.GEIZER, 
                font_size=font_size, 
                text_color=Colors.BLACK.value, 
                command=lambda: self.change_menu(self.sound_assets_menu)
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

    def render(self, screen):
        """
        Renders the settings menu scene, including the background, frame, labels, and buttons.
        
        Args:
            screen (pygame.Surface): The pygame screen object where the scene will be rendered.
        """
        screen.blit(self.bg, (0, 0))
        pygame.draw.rect(screen, Colors.DARK_GRAY.value, self.frame, border_radius=int(config.height*0.08))
        pygame.draw.rect(screen, Colors.WHITE.value, self.frame, width=1, border_radius=int(config.height*0.08))
        super().render(screen)
        self.volume_bar.draw(screen)
        pygame.draw.ellipse(screen, Colors.WHITE.value, self.volume_icon_rect)
        screen.blit(self.volume_icon, self.volume_icon_rect)
        if self.assets_menu:
            screen.blit(self.filter, (0, 0))
            self.assets_menu.render(screen)

    def update(self):
        if not self.assets_menu:
            super().update()
            self.volume_bar.update()
        else :
            self.assets_menu.update()
    
    def handle_event(self, event):
        """
        Handles events like mouse clicks to interact with buttons and adjust settings.
        
        Args:
            event (pygame.event): The event to be handled.
        """
        mouse_pos = pygame.mouse.get_pos()
        if not self.assets_menu:
            super().handle_event(event)
        else:
            self.assets_menu.handle_event(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if (mouse_pos[0] < list(self.assets_menu.buttons.values())[0].rect.left or mouse_pos[0] > list(self.assets_menu.buttons.values())[0].rect.right) and (not self.assets_menu.buttons['volume'].is_clicked() if type(self.assets_menu) == SoundAssetsMenu else True):
                    self.change_menu(None)
    
    def change_menu(self, menu:Scene):
        """
        Switches the displayed menu to one of the asset menus (piece, board, or sound).
        
        Args:
            menu (Scene): The menu to be displayed.
        """
        self.assets_menu = menu

class PieceAssetsMenu(Scene):
    """
    Represents the menu where the user can select and customize piece assets for the game.
    This menu allows users to choose different visual assets for game pieces, including a blindfold option.
    """

    def __init__(self, settings_menu):
        """
        Initializes the piece assets menu with a list of piece images, buttons, and a background image.
        
        Args:
            settings_menu (SettingsMenu): The settings menu that called this piece assets menu.
        """
        super().__init__()
        self.settings_menu = settings_menu
        self.piece_images = generate_piece_images()
        self.buttons[config.piece_asset].update_color(Colors.GREEN.value, None)
        self.board_image = generate_board_image()
        self.board_clip_rect = pygame.Rect(0, 0, config.tile_size*6, config.tile_size*2)

    def create_buttons(self):
        """
        Creates the buttons for each available piece asset option.
        
        The buttons allow the user to select different piece assets for the game.
        """
        button_width = int(config.width * 0.3)
        button_height = int(config.height * 0.1)
        font_size = int(button_height * 0.5)

        self.buttons = {
            asset: RectButton(
                x=config.width*0.2,
                y=int(config.height*0.05)+(button_height*i), 
                width=button_width, 
                height=button_height, 
                color=Colors.LIGHT_GRAY.value, 
                hovered_color=Colors.WHITE.value,
                text=asset, 
                font_name=Fonts.GEIZER, 
                font_size=font_size, 
                text_color=Colors.DARK_GRAY.value, 
            ) 
            for i, asset in enumerate(available_piece)
        }

    def create_labels(self):
        """
        Creates the labels to indicate the selected piece asset option.
        """
        self.labels = {'selected_asset': Label(
            (int(config.width*0.69), int(config.height*0.65)),
            config.piece_asset,
            Fonts.GEIZER,
            int(config.height*0.1),
            Colors.BLACK.value,
            create_rect_surface(Colors.WHITE.value, int(config.width*0.3), int(config.height*0.1), int(config.height*0.075)),
            (int(config.width*0.69), int(config.height*0.65))
        )}

    def render(self, screen):
        """
        Renders the piece assets menu scene, including the background, board image, and piece assets.

        Args:
            screen (pygame.Surface): The pygame screen object where the scene will be rendered.
        """
        super().render(screen)
        screen.blit(self.board_image, dest=(config.width*0.49, config.height*0.3), area=self.board_clip_rect)
        if config.piece_asset != 'blindfold':
            for i, piece in enumerate(self.piece_images.items()):
                notation, image = piece[0], piece[1]
                i = i % 6
                # Render white pieces at the top and black pieces at the bottom
                position = (config.width * 0.49 + (config.tile_size * i), config.height * 0.3)
                if notation[0] == 'w':
                    screen.blit(image, position)
                else:
                    screen.blit(image, (position[0], position[1] + config.tile_size))

    def handle_event(self, event):
        """
        Handles events like mouse clicks to interact with buttons and adjust the selected piece asset.

        Args:
            event (pygame.event): The event to be handled.
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                for asset, button in self.buttons.items():
                    if button.is_clicked():
                        self.change_asset(asset)
            elif event.button == 4 and list(self.buttons.values())[0].rect.top < 0:
                for button in self.buttons.values():
                    button.y += 25
                    button.rect.y += 25
                    button.label.rect.y += 25
            elif event.button == 5 and list(self.buttons.values())[len(self.buttons)-1].rect.bottom > config.height:
                for button in self.buttons.values():
                    button.y -= 25
                    button.rect.y -= 25
                    button.label.rect.y -= 25 

    def change_asset(self, asset):
        """
        Updates the currently selected piece asset and changes the button color accordingly.

        Args:
            asset (str): The selected piece asset to be applied.
        """
        self.buttons[config.piece_asset].update_color(Colors.LIGHT_GRAY.value, Colors.WHITE.value)
        config.piece_asset = asset
        self.buttons[asset].update_color(Colors.GREEN.value, None)
        if asset != 'blindfold':
            self.piece_images = generate_piece_images()
        self.labels['selected_asset'].update_text(config.piece_asset)


class BoardAssetsMenu(Scene):
    """
    Represents the menu where the user can select and customize the board assets for the game.
    This menu allows users to choose different visual assets for the game board.
    """

    def __init__(self, settings_menu):
        """
        Initializes the board assets menu with a list of available board assets, buttons, and a background image.

        Args:
            settings_menu (SettingsMenu): The settings menu that called this board assets menu.
        """
        super().__init__()
        self.settings_menu = settings_menu
        self.board_image = resize_image(generate_board_image(), (config.height*0.75, config.height*0.75))
        self.buttons[config.board_asset].update_color(Colors.GREEN.value, None) 

    def create_buttons(self):
        """
        Creates the buttons for each available board asset option.
        
        The buttons allow the user to select different board assets for the game.
        """
        button_width = int(config.width * 0.3)
        button_height = int(config.height * 0.1)
        font_size = int(button_height * 0.5)

        self.buttons = {
            asset: RectButton(
                x=config.width*0.2, 
                y=int(config.height*0.05)+(button_height*i), 
                width=button_width, 
                height=button_height, 
                color=Colors.LIGHT_GRAY.value, 
                hovered_color=Colors.WHITE.value,
                text=asset.replace('-', ' '), 
                font_name=Fonts.GEIZER, 
                font_size=font_size,
                text_color=Colors.DARK_GRAY.value, 
            ) 
            for i, asset in enumerate(available_board)
        }

    def create_labels(self):
        """
        Creates the labels to indicate the selected board asset option.
        """
        self.labels = {'selected_asset': Label(
            (int(config.width*0.69), int(config.height*0.9)),
            config.board_asset,
            Fonts.GEIZER,
            int(config.height*0.1),
            Colors.BLACK.value,
            create_rect_surface(Colors.WHITE.value, int(config.width*0.32), int(config.height*0.1), int(config.height*0.075)),
            (int(config.width*0.69), int(config.height*0.9))
        )}

    def render(self, screen):
        """
        Renders the board assets menu scene, including the background and the selected board asset image.

        Args:
            screen (pygame.Surface): The pygame screen object where the scene will be rendered.
        """
        super().render(screen)
        screen.blit(self.board_image, (config.width*0.47, config.margin))

    def handle_event(self, event):
        """
        Handles events like mouse clicks to interact with buttons and adjust the selected board asset.

        Args:
            event (pygame.event): The event to be handled.
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                for asset, button in self.buttons.items():
                    if button.is_clicked():
                        self.change_asset(asset)
            elif event.button == 4 and list(self.buttons.values())[0].rect.top < 0 :
                for button in self.buttons.values():
                    button.y += 25
                    button.rect.y += 25
                    button.label.rect.y += 25
            elif event.button == 5 and list(self.buttons.values())[len(self.buttons)-1].rect.bottom > config.height:
                for button in self.buttons.values():
                    button.y -= 25
                    button.rect.y -= 25
                    button.label.rect.y -= 25   

    def change_asset(self, asset):
        """
        Updates the currently selected board asset and changes the button color accordingly.

        Args:
            asset (str): The selected board asset to be applied.
        """
        self.buttons[config.board_asset].update_color(Colors.LIGHT_GRAY.value, Colors.WHITE.value)
        config.board_asset = asset
        self.buttons[asset].update_color(Colors.GREEN.value, None)
        self.board_image = resize_image(generate_board_image(), (config.height*0.75, config.height*0.75))
        self.settings_menu.piece_assets_menu.board_image = generate_board_image()
        self.labels['selected_asset'].update_text(config.board_asset)


class SoundAssetsMenu(Scene):
    """
    Represents the menu where the user can select and customize the sound assets for the game.
    This menu allows users to choose different sound assets for the game and control the volume.
    """

    def __init__(self, settings_menu):
        """
        Initializes the sound assets menu with a list of available sound assets, buttons for sound selection,
        and a button to play the selected sound.

        Args:
            settings_menu (SettingsMenu): The settings menu that called this sound assets menu.
        """
        super().__init__()
        self.settings_menu = settings_menu
        self.sounds = generate_sounds()
        self.sound_keys = list(self.sounds.keys())
        self.current_sound = 0
        self.buttons[config.sound_asset].update_color(Colors.GREEN.value, None)

    def create_buttons(self):
        """
        Creates the buttons for each available sound asset option, including a button to control volume.

        The buttons allow the user to select different sound assets and a volume button to adjust sound effects.
        """
        button_width = int(config.width * 0.3)
        button_height = int(config.height * 0.1)
        font_size = int(button_height * 0.5)

        self.buttons = {
            asset: RectButton(
                x=config.width*0.2, 
                y=int(config.height*0.1)+(button_height*i), 
                width=button_width, 
                height=button_height, 
                color=Colors.LIGHT_GRAY.value, 
                hovered_color=Colors.WHITE.value,
                text=asset, 
                font_name=Fonts.GEIZER, 
                font_size=font_size, 
                text_color=Colors.DARK_GRAY.value,
            ) 
            for i, asset in enumerate(available_sound)
        }
        self.buttons['volume'] = RectButton(
            x = config.width*0.69,
            y = config.height*0.43,
            width=config.height*0.2,
            height=config.height*0.2,
            color=Colors.LIGHT_GRAY.value,
            hovered_color=Colors.WHITE.value,
            border_radius=int(config.height*0.05),
            image=load_image('assets/images/volume.png', (config.height*0.15, config.height*0.15)),
            command=self.play_sound
        )

    def create_labels(self):
        """
        Creates the labels to indicate the selected sound asset option.
        """
        self.labels = {'selected_asset': Label(
            (int(config.width*0.69), int(config.height*0.63)),
            config.sound_asset,
            Fonts.GEIZER,
            int(config.height*0.1),
            Colors.BLACK.value,
            create_rect_surface(Colors.WHITE.value, int(config.width*0.32), int(config.height*0.1), int(config.height*0.075)),
            (int(config.width*0.69), int(config.height*0.63))
        )}

    def render(self, screen):
        """
        Renders the sound assets menu scene, including the background and the selected sound asset image.

        Args:
            screen (pygame.Surface): The pygame screen object where the scene will be rendered.
        """
        super().render(screen)

    def handle_event(self, event):
        """
        Handles events like mouse clicks to interact with buttons and adjust the selected sound asset.

        Args:
            event (pygame.event): The event to be handled.
        """
        super().handle_event(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                for asset, button in self.buttons.items():
                    if button.is_clicked() and asset != 'volume':
                        self.change_asset(asset)

    def change_asset(self, asset):
        """
        Updates the currently selected sound asset and changes the button color accordingly.

        Args:
            asset (str): The selected sound asset to be applied.
        """
        self.buttons[config.sound_asset].update_color(Colors.LIGHT_GRAY.value, Colors.WHITE.value)
        config.sound_asset = asset
        self.buttons[asset].update_color(Colors.GREEN.value, None)
        self.sounds = generate_sounds()
        self.current_sound = 0
        self.labels['selected_asset'].update_text(config.sound_asset)

    def play_sound(self):
        """
        Plays the currently selected sound asset. Cycles through the available sounds each time the button is clicked.
        """
        self.sounds[self.sound_keys[self.current_sound]].play()
        self.current_sound = (self.current_sound + 1) % len(self.sounds)

    def update_volume(self):
        """
        Updates the volume for all sounds according to the current volume setting.

        This method is called to apply the global volume setting to all sound assets.
        """
        for sound in self.sounds.values():
            sound.set_volume(config.volume)


class VolumeBar:
    """
    Represents a volume control bar that allows users to adjust the volume of the game.
    It visually displays the volume level and allows for interaction via mouse input.
    """

    def __init__(self):
        """
        Initializes the volume control bar with a rectangle representing the bar's position and size.
        The volume bar is positioned based on the screen dimensions.

        The default position and size are set relative to the window's width and height.
        """
        self.rect = pygame.Rect(config.width*0.4, config.height * 0.7, config.width * 0.25, config.height * 0.07)

    def draw(self, screen):
        """
        Renders the volume bar on the screen. The bar is drawn as a background rectangle 
        with a filled rectangle indicating the current volume level.

        Args:
            screen (pygame.Surface): The screen surface where the volume bar will be rendered.
        """
        pygame.draw.rect(screen, Colors.GRAY.value, self.rect)
        filled_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width * config.volume, self.rect.height)
        pygame.draw.rect(screen, Colors.WHITE.value, filled_rect)

    def update(self):
        """
        Updates the volume level based on mouse interaction. If the user clicks and drags the mouse
        on the volume bar, the volume is adjusted accordingly, constrained between 0 and 1.

        The volume level is updated only when the left mouse button is pressed and the mouse is within the bounds of the bar.
        """
        if pygame.mouse.get_pressed()[0]:
            mouse_pos = pygame.mouse.get_pos()
            if self.rect.collidepoint(mouse_pos):
                config.volume = (mouse_pos[0] - self.rect.x) / self.rect.width
                config.volume = max(0, min(1, config.volume))  # Constrain the volume between 0 and 1
