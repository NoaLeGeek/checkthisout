import os
import sys
import configparser


class Config:
    def __init__(self):
        """
        Initializes the configuration object by reading settings from a configuration file.

        This constructor uses the `configparser` module to parse a configuration file (`config.cfg`) 
        and initializes various attributes of the object based on the file's contents. 
        These attributes include general settings, asset paths, board dimensions, debug mode, 
        and a dictionary of chess rule variations.

        Attributes:
            config (ConfigParser): The configuration parser object used to read the configuration file.
            fps (int): Frames per second setting for the application.
            volume (float): Volume level for the application.
            piece_asset (str): Path to the chess piece assets.
            board_asset (str): Path to the chessboard assets.
            sound_asset (str): Path to the sound assets.
            flipped_assets (bool): Whether the assets are flipped.
            background_asset (str): Path to the background asset.
            rows (int): Number of rows on the chessboard.
            columns (int): Number of columns on the chessboard.
            debug (bool): Debug mode flag.
            rules (dict): A dictionary of chess rule variations, where each key is a rule name 
                          (str) and the value is a boolean indicating whether the rule is enabled.
        """
        self.config = configparser.ConfigParser()
        self.config.read('config.cfg')
        self.fps = self.config.getint('GENERAL', 'fps')
        self.volume = self.config.getfloat('GENERAL', 'volume')
        self.piece_asset = self.config.get('ASSETS', 'piece')
        self.board_asset = self.config.get('ASSETS', 'board')
        self.sound_asset = self.config.get('ASSETS', 'sound')
        self.flipped_assets = self.config.getboolean('ASSETS', 'flipped_assets')
        self.background_asset = self.config.get('ASSETS', 'background')
        self.rows = self.config.getint('BOARD', 'rows')
        self.columns = self.config.getint('BOARD', 'columns')
        self.debug = self.config.getboolean('GENERAL', 'debug')
        self.rules = {
            "classic": True,
            "puissance_4_pawns": False,
            "mutation_chess": False,
            "king_of_the_hill": False,
            "+3_checks": False,
            "giveaway": False,
            "chess960": False,
            "random_position": False,
            "random_promotion": False,
            "no_promotion": False,
            "only_piece_promotion": False, # types of pieces
            "no_en_passant": False,
            "en_passant_forced": False,
            "swapper": False,  # n pourcentage
            "freezer": False,  # n pourcentage
            "any_capture": False,
            "barbarians": False,  # n pourcentage and types of pieces
            "torpedo": False,
            "crawl": False,
            "overpass": False,
            "retreat": False,
            "leap": False,
            "sideways": False,
            "upside_down": False,
            "regal_sweep": False,
            "frozen_wonderland": False,
            "reverse_strike": False,
            "inward_strike": False,
            "allow_passing": False,
            "double_play": False,
            "bare_piece": False,
            "duck_chess": False,
            "opposite_castling": False,
            "no_big_castle": False,
            "no_small_castle": False,
            "no_castling": False,
            "ignore_check": False,
            "teleporter_madness": False,
            "atomic": False,
            "fatal_capture": False,
            "blindfold": False,
            "ghostboard": False,
            "forward_march": False,
            "colorblind": False,
            "crazyhouse": False,
            "racing_kings": False,
            "4_player_chess": False,
            "mystery_chess": False,
            "royal_quota": False,
            "corregal": False,
            "decimation": False,
            "knightmate": False,
            "extinction": False,
            "pawns_only": False,
            "cylinder": False,
            "bouncing": False,
            "infinite_bouncing": False,
            "berolina": False,
            "absorption": False,
            "cannibal": False,
            "fog_of_war": False,
            "chess18": False,
            "shuffle_chess": False,  # no castling
            "chess2880": False,
            "reverse_backrank": False,
            "random_play_first": False,
            "black_play_first": False,
            "half_chaos": False,
            "random_asset": False,
            "asset_chaos": False,
            "thanos": False,  # n pourcentage
            "mirror": False,
            "pawn_parade": False,
            "half_commitment": False,
            "unique_step": False,
            "instant_void": False,  # n moves
            "random_walls": False,  # n pourcentage
            "minefield": False,  # n pourcentage
            "super_size": False,  # n
            "abstract": False,
            "accelerated": False,
            "active": False,
            "rotating_center": False,  # clockwise, anticlockwise or diagonal
            "advance": False,
            "el_vaticano": False,
            "all-in": False,
            "all-mate": False,
            "almost": False,
            "amazon": False,
            "annihilation": False,
            "anti-gravity": False,
            "anti-magnet": False,
            "anywhere": False,
            "archimedes": False,
            "alternating": False,
            "atlantis": False,
            "bachelor": False,
        }

    # Not implemented
    def save(self):
        pass

    def set_dimensions(self, width, height):
        """
        Sets the dimensions and calculates related properties for a chess game board.

        This method adjusts the height of the board to maintain a 16:9 aspect ratio 
        based on the given width. It also calculates the margin, tile size, and 
        evaluation bar width based on the board's dimensions and the number of columns.

        Parameters:
            width (int): The width of screen.
            height (int): The height of the screen. This value is recalculated 
                          internally to maintain a 16:9 aspect ratio.

        Attributes:
            self.width (int): The width of the chess game board.
            self.height (int): The height of the chess game board, adjusted to maintain 
                               a 16:9 aspect ratio.
            self.margin (int): The margin size around the board, calculated based on 
                               the height and number of columns.
            self.tile_size (int): The size of each tile on the board, calculated based 
                                  on the height and number of columns.
            self.eval_bar_width (int): The width of the evaluation bar, calculated as 
                                       half the tile size.
        """
        self.width = width
        self.height = int(self.width*(9/16))
        while self.height > height :
            self.width -= 1
            self.height = int(self.width*(9/16))
        self.margin = self.height//(self.columns*2 + 2)
        self.tile_size = self.height//(self.columns + 1)
        self.eval_bar_width = self.tile_size//2

    def update_rule(self, rule, value = None):
        """
        Updates the value of a specific rule in the rules dictionary.

        If a value is provided, the rule will be set to that value. If no value is provided,
        the method will toggle the current value of the rule (i.e., switch between True and False).

        Parameters:
            rule (str): The name of the rule to update.
            value (bool, optional): The new value to set for the rule. If not provided, the rule's
                                    value will be toggled.
        """
        if value is None:
            value = not bool(self.rules.get(rule))
        self.rules[rule] = value

config = Config()