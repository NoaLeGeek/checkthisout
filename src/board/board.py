from random import choice
from math import ceil

import pygame
import numpy as np

from src.gui import Label
from src.config import config
from src.board.tile import Tile
from src.board.player import Player
from src.ia.negamax import NegamaxAI
from src.board.move import Move, MoveTree
from src.constants import castling_king_column, en_passant_direction, Fonts, Colors
from src.board.piece import notation_to_piece, piece_to_notation, piece_to_num
from src.utils import generate_piece_images, generate_board_image, generate_sounds, flip_pos, play_sound

class Board:
    """Initializes a new Board instance using the given players and optional FEN string.

        Sets up the internal representation of the chess board, assigns players, initializes
        gameplay attributes (turn, castling rights, en passant target, move counters, etc.),
        and loads external resources such as piece images and sounds. If the current player is
        an AI, the AI makes its move immediately after initialization.

        Args:
            current_player (Player): The player whose turn it is to move.
            waiting_player (Player): The opponent player.
            fen (str, optional): A FEN string describing the initial board state. Defaults to
                standard starting position.

        Attributes:
            board (dict): A dictionary mapping board coordinates to Tile objects.
            selected (Optional[Tuple[int, int]]): The currently selected square.
            turn (int): Indicates the current turn (1 for white, -1 for black).
            winner (Optional[int]): Indicates the winner (1 for white, -1 for black, or None).
            en_passant (Optional[Tuple[int, int]]): The square available for en passant capture.
            promotion (Optional[Type[Piece]]): The piece type to which a pawn is being promoted.
            half_moves (int): Counter for the fifty-move rule.
            full_moves (int): Number of full moves in the game.
            flipped (int): Orientation of the board (1 or -1).
            last_irreversible_move (int): Move number of the last irreversible move.
            game_over (bool): Indicates if the game is over.
            current_player (Player): The player whose turn it is.
            waiting_player (Player): The opposing player.
            castling (dict): Dictionary tracking castling rights per color and direction.
            score (int): Evaluation score of the current board position.
            negamax (NegamaxAI): AI engine for evaluating the board.
            checks (dict): Tracks number of checks in +3 checks rule mode (only if enabled).
            image (Surface): Graphical representation of the board background.
            piece_images (dict): Dictionary of images for each piece.
            sounds (dict): Dictionary of sounds for different move types.
            move_tree (MoveTree): Tree structure storing the game history.
            history (List[str]): List of move notations.
            history_change (bool): Flag indicating whether the history was updated.

        Raises:
            ValueError: If the FEN string is malformed.
        """    
    def __init__(self, current_player: Player, waiting_player: Player, fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
        # Initialize board attributes
        self.board = {}
        self.selected = None
        self.turn = 1
        self.winner = None
        self.en_passant = None
        self.promotion = None
        self.half_moves = 0
        self.full_moves = 1
        self.flipped = 1
        self.last_irreversible_move = 0
        self.game_over = False
        self.current_player = current_player
        self.waiting_player = waiting_player
        self.castling = {1: {1: False, -1: False}, -1: {1: False, -1: False}}
        self.score = 0
        self.negamax = NegamaxAI(0, 0)

        # Anarchy chess
        if config.rules["+3_checks"] == True:
            self.checks = {1: 0, -1: 0}

        # Load resources
        self.image = generate_board_image()
        self.piece_images = generate_piece_images(self.flipped)
        self.sounds = generate_sounds()

        # Initialize the board from the FEN string
        self.move_tree = MoveTree(self)

        self.history_change = False
        self.history = []
        self._create_board(fen)

        # IA plays first if the white player is an IA
        if self.current_player.ia == True:
            self.current_player.play_move(self)

    def _create_board(self, fen: str) -> None:
        """
        Initializes the board based on the provided FEN string.

        This function sets up the chessboard by parsing the given FEN (Forsyth-Edwards Notation)
        string, handling chess960 if applicable, and initializing various board parameters 
        such as the pieces, turn, castling rights, en passant, half moves, and full moves.

        Args:
            fen (str): The FEN string representing the game state. It must contain exactly 6 parts:
                - The piece placement (8 rows of 8 characters representing the board).
                - The active color ('w' for white or 'b' for black).
                - The castling availability.
                - The en passant target square, if any.
                - The halfmove clock (for fifty-move rule).
                - The fullmove number.

        Raises:
            ValueError: If the FEN string is not properly formatted or contains invalid data.
        """
        self.board = {(row, column): Tile((row, column)) for row in range(config.rows) for column in range(config.columns)}
        try:
            # Chess960 row generation
            if config.rules["chess960"] == True:
                fen = self._transform_960_fen(fen)

            fen_parts = fen.split()
            if len(fen_parts) != 6:
                raise ValueError("Invalid FEN format: Must contain exactly 6 parts.")

            # Initialize board state
            self.turn = 1 if fen_parts[1] == "w" else -1
            self._initialize_pieces(fen_parts[0])
            self._initialize_castling(fen_parts[2])
            self.en_passant = self._parse_en_passant(fen_parts[3])
            self.half_moves = int(fen_parts[4])
            self.full_moves = int(fen_parts[5])
            play_sound(self.sounds, "game-start")
        except (IndexError, ValueError) as e:
            raise ValueError(f"Failed to parse FEN string: {fen}. Error: {e}")

    def _initialize_pieces(self, board_part: str):
        """
        Initializes the chess pieces on the board based on the board part from the FEN string.

        This function parses the part of the FEN string that represents the piece placement on the board,
        creating and placing each piece in the correct location. It also assigns images to the pieces and tracks 
        the positions of the kings for both players.

        Args:
            board_part (str): A string representing the layout of the pieces on the board. 
                            The string is divided by slashes ('/'), where each part represents a row of the board.
                            Empty squares are represented by numbers (1-8), indicating how many squares are empty.

        Raises:
            ValueError: If an invalid piece notation is encountered or if there are missing piece images.
        """
        for row, part in enumerate(board_part.split("/")):
            column = 0
            for char in part:
                if char.isdigit():
                    column += int(char)  # Skip empty squares
                else:
                    color = 1 if char.isupper() else -1
                    piece_type = notation_to_piece(char)
                    if not piece_type:
                        raise ValueError(f"Invalid piece notation: {char}")
                    piece = piece_type(color)
                    if config.piece_asset != "blindfold":
                        piece_image_key = f"{(("w" if color == 1 else "b") if config.piece_asset != "mono" else "")}{char.upper()}"
                        if piece_image_key not in self.piece_images:
                            raise ValueError(f"Missing piece image for: {piece_image_key}")
                        piece.image = self.piece_images[piece_image_key]
                    self.get_player(color).add_piece(piece)
                    self.board[(row, column)].piece = piece
                    # Track the king's position
                    if piece.notation == "K":
                        self.get_player(color).king_pos = (row, column)
                    column += 1

    def _initialize_castling(self, castling_part: str):
        """
        Initializes the castling rights for both players based on the castling part of the FEN string.

        This function parses the castling information from the FEN string to set up the castling rights for both 
        the white and black players. Castling rights are tracked separately for each player and each direction (King-side and Queen-side).

        Args:
            castling_part (str): A string representing the castling rights. Possible values include:
                - "-" indicates no castling rights.
                - "K" (King-side) or "Q" (Queen-side) for the white player.
                - "k" (King-side) or "q" (Queen-side) for the black player.
                Castling rights for both players are represented by the presence of "K" or "Q" (uppercase for white, lowercase for black).
        """
        if castling_part == "-":
            self.castling = {1: {1: False, -1: False}, -1: {1: False, -1: False}}
            return

        for color in [1, -1]:
            for direction, letter in zip([1, -1], ["K", "Q"]):
                letter = letter.lower() if color == -1 else letter
                if letter in castling_part:
                    rook_position = self._find_rook_for_castling(color, direction)
                    if rook_position:
                        self.castling[color][direction] = True

    def _find_rook_for_castling(self, color: int, direction: int) -> bool:
        """
        Finds the rook needed for castling in the specified direction.

        This function checks if the rook that can be used for castling is present in the row of the king. 
        It searches in the direction of the castling (either King-side or Queen-side) to find a rook of the 
        specified color, starting from the king's position.

        Args:
            color (int): The color of the player (1 for white, -1 for black).
            direction (int): The direction of the castling (-1 for Queen-side, 1 for King-side).

        Returns:
            bool: Returns `True` if a rook of the correct color is found in the castling direction, 
                `False` otherwise.
        """
        row, king_col = self.get_player(color).king_pos
        for col in range(king_col, -1 if direction == -1 else config.columns, direction):
            piece = self.get_piece((row, col))
            if piece and piece.notation == "R" and piece.color == color:
                return True
        return False

    def _parse_en_passant(self, en_passant_part: str):
        """
        Parses the en passant notation from the FEN string and returns the corresponding board position.

        This function converts the en passant notation (e.g., "e3" or "d6") into a row and column index
        on the board. If the notation is invalid or indicates no en passant available, the function handles
        those cases appropriately.

        Args:
            en_passant_part (str): The en passant notation from the FEN string. It can be a letter-number 
                                combination (e.g., "e3") representing the target square, or "-" or "–" 
                                if there is no en passant.

        Returns:
            tuple or None: Returns a tuple (row, col) representing the en passant target square, or 
                            `None` if there is no en passant.

        Raises:
            ValueError: If the en passant notation is invalid.
        """
        if en_passant_part in ['-', '–']:
            return None
        if len(en_passant_part) != 2 or not en_passant_part[0].isalpha() or not en_passant_part[1].isdigit():
            raise ValueError(f"Invalid en passant notation: {en_passant_part}")

        row = flip_pos(int(en_passant_part[1]) - 1, flipped=-self.flipped)
        col = flip_pos(ord(en_passant_part[0]) - ord('a'), flipped=self.flipped)
        return row, col

    def _transform_960_fen(self, fen: str):
        """
        Transforms a standard FEN string into a Chess960 (Fischer Random Chess) FEN string.

        This function modifies the FEN string to place the back rank pieces in a random but legal order, 
        according to the Chess960 setup rules. Bishops are placed on opposite-colored squares, and the other 
        pieces (knights, queen, rooks, and king) are placed in random positions on the back rank, ensuring 
        all rules for Chess960 are followed.

        Args:
            fen (str): The original FEN string representing the board state.

        Returns:
            str: The transformed FEN string representing the Chess960 setup.
        """
        # Initialize the row
        last_row = [None] * config.columns

        # Place the bishops on opposite-colored squares
        light_square_indices = list(range(0, config.columns, 2))
        dark_square_indices = list(range(1, config.columns, 2))
        last_row[choice(light_square_indices)] = "B"
        last_row[choice(dark_square_indices)] = "B"

        # Place the remaining pieces: knights, queen, rooks, and king
        pieces = ["N", "N", "Q"]
        empty_indices = [i for i, val in enumerate(last_row) if val is None]
        for piece in pieces:
            selected_index = choice(empty_indices)
            last_row[selected_index] = piece
            empty_indices.remove(selected_index)
        for piece, index in zip(["R", "K", "R"], sorted(empty_indices)):
            last_row[index] = piece

        fen_parts = fen.split()

        # Update the FEN string for the board
        rows = fen_parts[0].split("/")
        for row in [0, 7]:
            rows[row] = "".join(last_row).lower() if row == 0 else "".join(last_row)
        fen_parts[0] = "/".join(rows)
        return " ".join(fen_parts)
    
    def check_game(self):
        """
        Checks the game state to determine if a game-ending condition has been met.

        This method evaluates various win, loss, or draw conditions based on the game configuration.
        It checks for specific conditions such as "King of the Hill," "+3 Checks," "Giveaway," 
        stalemate, the 50-move rule, insufficient material, and threefold repetition. If any of these 
        conditions are met, the game state is updated, and the winner is determined.

        If a game-ending condition is met, the `winner` attribute is updated, and the `game_over` 
        attribute is set to `True`. A sound is also played to signal the end of the game.

        Attributes:
            winner (str or None): The winner of the game, or a string representing the draw condition 
                                (e.g., "Stalemate", "Draw by the 50-move rule"). `None` if the game is ongoing.
            game_over (bool): A flag indicating if the game has ended.

        Raises:
            None
        """
        if config.rules["king_of_the_hill"] == True and self.waiting_player.king_pos in self.get_center():
            self.winner = "Black" if self.turn == 1 else "White"
        elif config.rules["+3_checks"] == True and self.checks[-self.turn] >= 3:
            self.winner = "Black" if self.turn == 1 else "White"
        elif config.rules["giveaway"] == True and self.waiting_player.pieces == {}:
            self.winner = "Black" if self.turn == 1 else "White"
        elif self.is_stalemate():
            if self.current_player.is_king_check(self) or config.rules["giveaway"] == True:
                self.winner = "Black" if self.turn == 1 else "White"
            else:
                self.winner = "Stalemate"
        elif self.half_moves >= 100:
            self.winner = "Draw by the 50-move rule"
        elif self.is_insufficient_material():
            self.winner = "Draw by insufficient material"
        elif self.is_threefold_repetition():
            self.winner = "Draw by threefold repetition"
        if self.winner is not None:
            self.game_over = True
            play_sound(self.sounds, "game-end")

    def is_threefold_repetition(self):
        """
        Checks if a threefold repetition has occurred in the game.

        This method analyzes the FEN positions from the last irreversible move to the current position. 
        It counts how many times each position has occurred and determines if any position has been 
        repeated three or more times, which would indicate a draw by threefold repetition.

        Returns:
            bool: `True` if a threefold repetition has occurred, otherwise `False`.
        """
        positions = [move.fen.split(" ")[:4] for move in self.move_tree.get_root_to_leaf()[self.last_irreversible_move:]]
        return any(positions.count(pos) >= 3 for pos in positions)

    def is_stalemate(self):
        """
        Checks if the current game state is a stalemate.

        A stalemate occurs when the current player has no legal moves left and their king is not in check.

        Returns:
            bool: `True` if the current player is in stalemate (no legal moves), otherwise `False`.
        """
        return len(self.current_player.get_legal_moves(self)) == 0

    def is_insufficient_material(self):
        """
        Checks if the game is in a state of insufficient material to checkmate.

        This method evaluates the number and types of pieces remaining on the board to determine if either 
        player has insufficient material to force a checkmate. It considers the following scenarios:
        - Only kings remain (2 pieces).
        - One player has a king and a single bishop.
        - Both players have only a king and a single bishop of the same color.

        Returns:
            bool: `True` if there is insufficient material for either player to checkmate, otherwise `False`.
        """
        piece_count = self.count_pieces()
        # Only kings remain
        if piece_count == 2:
            return True  
        if piece_count == 3:
            return any(
                len(self.get_player(color).pieces.get("B")) == 1 or
                len(self.get_player(color).pieces.get("K")) == 1
                for color in [-1, 1]
            )
        if piece_count == 4:
            if all(len(self.get_player(color).pieces.get("B")) == 1 for color in [-1, 1]):
                square_colors = [self.find_tile("B", color).get_square_color() for color in [-1, 1]]
                if square_colors[0] == square_colors[1]:
                    return True
        return False
    
    def count_pieces(self):
        """
        Counts the total number of pieces for both the current player and the waiting player.

        This method returns the sum of the pieces for both players by calling `count_pieces` on both 
        the current player and the waiting player.

        Returns:
            int: The total number of pieces for both players combined.
        """
        return self.current_player.count_pieces() + self.waiting_player.count_pieces()
    
    def find_tile(self, notation, color):
        """
        Finds the tile containing a specific piece based on its notation and color.

        This method searches through all the tiles on the board and returns the first tile that contains 
        a piece matching the specified notation (e.g., "K" for king, "Q" for queen) and color. If no such 
        piece is found, it returns `None`.

        Args:
            notation (str): The notation of the piece to find (e.g., "K" for king, "Q" for queen).
            color (int): The color of the piece to find (1 for white, -1 for black).

        Returns:
            Tile or None: The tile containing the specified piece, or `None` if no such piece is found.
        """
        return next((tile for tile in self.board.values() if tile.piece and tile.piece.notation == notation and tile.piece.color == color), None)
    
    def convert_to_move(self, from_pos, to_pos, promotion=None):
        """
        Converts the given positions into a `Move` object.

        This method creates a `Move` object representing a move from a starting position (`from_pos`) to 
        a destination position (`to_pos`). An optional promotion piece can also be provided if the move 
        involves a pawn promotion.

        Args:
            from_pos (tuple): The starting position of the piece as a tuple (row, column).
            to_pos (tuple): The destination position of the piece as a tuple (row, column).
            promotion (str, optional): The type of piece to promote a pawn to (e.g., "Q" for queen). 
                                    Defaults to `None` if no promotion is required.

        Returns:
            Move: A `Move` object representing the move from `from_pos` to `to_pos` with an optional promotion.
        """
        return Move(self, from_pos, to_pos, promotion)

    def get_tile(self, pos: tuple[int, int]):
        """
        Retrieves the tile at a specific position on the board.

        This method looks up a tile at the given position (`pos`) on the board. If the position is valid, 
        it returns the corresponding tile; otherwise, it returns `None`.

        Args:
            pos (tuple[int, int]): The position of the tile on the board, represented as a tuple (row, column).

        Returns:
            Tile or None: The tile at the given position, or `None` if the position is invalid.
        """
        return self.board.get(pos, None)
    
    def get_piece(self, pos: tuple[int, int]):
        """
        Retrieves the piece at a specific position on the board.

        This method fetches the piece located at the given position (`pos`). If the position is valid but 
        does not contain a piece, or if the position is invalid, an error is raised.

        Args:
            pos (tuple[int, int]): The position of the tile on the board, represented as a tuple (row, column).

        Returns:
            Piece: The piece at the given position.

        Raises:
            ValueError: If the position is invalid or does not contain a piece.
        """
        try:
            return self.get_tile(pos).piece
        except AttributeError:
            raise ValueError(f"Invalid position: {pos}")
    
    def is_empty(self, pos):
        """
        Checks if the tile at a specific position is empty (i.e., contains no piece).

        This method determines if the tile at the given position (`pos`) does not have a piece by calling 
        `get_piece` and checking if the result is `None`.

        Args:
            pos (tuple[int, int]): The position of the tile on the board, represented as a tuple (row, column).

        Returns:
            bool: `True` if the tile at the given position is empty, otherwise `False`.
        """
        return self.get_piece(pos) is None

    def _update_castling(self, move: Move):
        """
        Updates castling rights based on the given move.

        This method modifies the castling rights for a player if the move involves a king or rook. 
        - If the king moves, all castling rights for that player are revoked.
        - If a rook moves, the castling right for the corresponding side (kingside or queenside) is revoked.

        Args:
            move (Move): The move being executed. Contains the moving piece and its origin position.
        """
        piece = move.moving_piece
        if piece.notation == "K":
            # If the King moves, reset castling rights for that player
            self.castling[piece.color] = {1: False, -1: False}
        elif piece.notation == "R":
            # If the Rook moves, update the castling rights for that rook's side
            side = 1 if move.from_pos[1] > self.current_player.king_pos[1] else -1
            self.castling[piece.color][side] = False

    def _update_last_irreversible_move(self, move: Move):
        """
        Updates the index of the last irreversible move in the move history.

        This method determines whether the given move is irreversible based on the following criteria:
        - The move is a capture.
        - The moving piece is a pawn.
        - The move involves castling.
        - The move results in a change to castling rights.

        If any of these conditions are met, the method updates `self.last_irreversible_move` to the 
        current length of the move sequence from root to leaf in the move tree.

        Args:
            move (Move): The move to evaluate for irreversibility.
        """
        if move.is_capture() or move.moving_piece.notation == "P" or move.castling or self.move_tree.current.castling != self.castling:
            # If the move is a capture, pawn move, castling, or a change in castling rights, mark it as irreversible
            self.last_irreversible_move = len(self.move_tree.get_root_to_leaf())

    def _is_valid_en_passant(self, pos: tuple[int, int], en_passant: tuple[int, int]):
        """
        Determines whether an en passant capture is valid from the given position.

        This method checks if any pawn adjacent to the given position (`pos`) can legally perform 
        an en passant capture to the specified target square (`en_passant`). The method validates:
        - The adjacent square is within bounds and contains an opponent pawn.
        - The pawn is eligible to capture en passant based on its direction and color.
        - The resulting move is legal in the current game state.

        Args:
            pos (tuple[int, int]): The reference position for checking possible en passant captures.
            en_passant (tuple[int, int]): The target position for the en passant capture.

        Returns:
            bool: `True` if a valid en passant capture exists, otherwise `False`.
        """
        d_ep = en_passant_direction[en_passant[0]]
        for d_col in [-1, 1]:
            new_pos = (pos[0], pos[1] + d_col)
            if not self.in_bounds(new_pos) or self.is_empty(new_pos):
                continue
            piece = self.get_piece(new_pos)
            if piece.notation != "P" or piece.color != d_ep*self.flipped:
                continue
            if self.convert_to_move(new_pos, en_passant).is_legal(self):
                return True
        return False

    def _update_en_passant(self, move):
        """
        Updates the en passant square based on the last move.

        This method sets the `en_passant` attribute to the appropriate square
        if a pawn has moved two squares forward from its starting position, 
        and the move qualifies for an en passant capture.

        Args:
            move (Move): The move that was just played, containing the starting 
                and ending positions of the piece.
        """
        from_pos, to_pos = move.from_pos, move.to_pos
        self.en_passant = None
        if not self.is_empty(from_pos) and self.get_piece(from_pos).notation == "P" and abs(from_pos[0] - to_pos[0]) == 2:
            en_passant = ((from_pos[0] + to_pos[0]) // 2, from_pos[1])
            if self._is_valid_en_passant((to_pos[0], to_pos[1]), en_passant):
                self.en_passant = en_passant
        
    def select(self, pos: tuple[int, int]):
        """
        Handles the selection of a square on the board and processes move logic.

        If a piece is already selected, this method attempts to perform various checks 
        including promotion, ally selection, deselection, illegal move handling, and 
        move execution. If no piece is selected, it attempts to select the piece 
        at the given position.

        Args:
            pos (tuple[int, int]): The position on the board being selected, 
                represented as (row, column).
        """
        if self.selected is not None:
            if self._trigger_promotion(pos):
                return
            if self._ally_piece(pos):
                return
            if self._deselect_piece(pos):
                return
            if self._handle_illegal_move(pos):
                return
            if self._set_promotion(pos):
                return
            self.convert_to_move(self.selected.pos, pos).execute(self)
        else:
            self._select_piece(pos)

    def _trigger_promotion(self, pos):
        """
        Attempts to trigger a promotion based on the selected position.

        If a pawn is eligible for promotion and the promotion state is active,
        this method checks if the clicked position corresponds to one of the 
        promotion options. If so, it executes the promotion move. If the click 
        is outside the promotion range, the promotion is canceled.

        Args:
            pos (tuple[int, int]): The position selected by the player.

        Returns:
            bool: True if a promotion was triggered or canceled, False otherwise.
        """
        if self.selected.piece.notation == "P" and self.promotion is not None:
            d = self.selected.piece.color * self.flipped
            if pos[0] in range(flip_pos(0, flipped=d), flip_pos(0, flipped=d) + d*len(self.selected.piece.promotion), d) and pos[1] == self.promotion[1]:
                self.convert_to_move(self.selected.pos, self.promotion, self.selected.piece.promotion[flip_pos(pos[0], flipped=d)]).execute(self)
                return True
            # Cancel promotion if the player doesn't click in the range of promotion
            self.promotion = None
            self.selected = None
            return True
        return False

    def _ally_piece(self, pos):
        """
        Handles selection logic when the selected position contains an allied piece.

        If the position contains an allied piece different from the currently 
        selected one, this method checks for a castling move if the selected piece 
        is a king and the target is a rook. Otherwise, it deselects the current 
        piece and selects the new allied piece.

        Args:
            pos (tuple[int, int]): The position selected by the player.

        Returns:
            bool: True if the position contained an allied piece and an action was taken,
            False otherwise.
        """
        if not self.is_empty(pos) and self.get_piece(pos).is_ally(self.selected.piece) and pos != self.selected.pos:
            # Castling move
            if self.selected.piece.notation == "K" and not self.is_empty(pos) and self.get_piece(pos).notation == "R" and pos in self.selected.calc_moves(self):
                self.convert_to_move(self.selected.pos, pos).execute(self)
                return True
            self.selected = None
            self.select(pos)
            return True
        return False

    def _deselect_piece(self, pos):
        """
        Deselects the currently selected piece if the given position matches its position
        or if the selected position is out of bounds.

        Args:
            pos (tuple[int, int]): The position selected by the player.

        Returns:
            bool: True if the piece was deselected, False otherwise.
        """
        if pos == self.selected.pos or not self.in_bounds(pos):
            self.selected = None
            return True
        return False

    def _handle_illegal_move(self, pos):
        """
        Handles scenarios where the selected move is illegal.

        If the target position is not in the list of legal moves for the selected piece,
        the selection is cleared. If the current player's king is in check, an 
        "illegal" sound is played. If the position contains a piece of the current
        player, it becomes the new selection.

        Args:
            pos (tuple[int, int]): The attempted move destination.

        Returns:
            bool: True if the move was illegal and handled, False otherwise.
        """
        if pos not in [move.to_pos for move in self.selected.piece.moves]:
            self.selected = None
            if self.current_player.is_king_check(self):
                play_sound(self.sounds, "illegal")
            if not self.is_empty(pos) and self.get_piece(pos).color == self.turn:
                self.select(pos)
            return True
        return False
    
    def _set_promotion(self, pos):
        """
        Sets the promotion state if a pawn reaches the final rank.

        Checks if the selected piece is a pawn and whether it has moved to the 
        promotion rank (first or last row). If so, marks the position for promotion.

        Args:
            pos (tuple[int, int]): The position the pawn has moved to.

        Returns:
            bool: True if the promotion state was set, False otherwise.
        """
        if self.selected.piece.notation == "P" and pos[0] in [0, config.rows - 1]:
            self.promotion = pos
            return True
        return False

    def _select_piece(self, pos):
        """
        Selects a piece at the given position if it belongs to the current player.

        If the position is occupied by a piece of the current player's color, 
        the piece is selected, and its available moves are filtered and stored.

        Args:
            pos (tuple[int, int]): The position of the piece to be selected.
        """
        if self.is_empty(pos) or self.get_piece(pos).color != self.turn:
            return
        self.selected = self.get_tile(pos)
        self.selected.piece.moves = self._filter_moves(self.selected)

    def _filter_moves(self, tile):
        """
        Filters the available moves for a given piece based on the current game rules.

        This method generates all possible moves for the piece, then applies additional
        filtering based on whether the "giveaway" rule is enabled or not. If enabled, 
        only capture moves are considered, otherwise only legal moves are kept.

        Args:
            tile (Tile): The tile containing the piece for which the moves are being filtered.

        Returns:
            list[Move]: A list of valid moves for the selected piece after filtering.
        """
        moves = [self.convert_to_move(tile.pos, move) for move in tile.piece.calc_moves(self, tile.pos)]
        if config.rules["giveaway"] == True:
            if len([move for move in self.current_player.get_moves(self) if move.is_capture()]) != 0:
                return [move for move in moves if move.is_capture()]
            return [move for move in moves if not move.castling]
        else:
            return [move for move in moves if move.is_legal(self)]

    def in_bounds(self, pos: tuple[int, int]) -> bool:
        """
        Checks if the given position is within the bounds of the board.

        This method verifies whether the position corresponds to a valid tile 
        on the board.

        Args:
            pos (tuple[int, int]): The position to check, represented as (row, column).

        Returns:
            bool: True if the position is within the board's bounds, False otherwise.
        """
        return self.get_tile(pos) is not None
    
    def flip_board(self) -> None:
        """
        Flips the board, reversing the orientation and updating various elements.

        This method flips the board tiles, updates the state of the currently selected 
        piece, and handles the flipping of player kings, the en passant square, and the 
        move tree. It also regenerates the piece images based on the flipped state 
        and updates the board image visually.

        Args:
            None

        Returns:
            None
        """
        self._flip_board_tiles()
        self.flipped *= -1
        # Remove the highlight of the selected piece
        if self.selected is not None:
            self.selected.highlight_color = None
            self.selected = None
        self.promotion = None
        # Flipping the kings' positions
        for color in [1, -1]:
            player = self.get_player(color)
            player.king_pos = flip_pos(player.king_pos)
        # Flipping the en passant square
        if self.en_passant:
            self.en_passant = flip_pos(self.en_passant)
        # Flipping the move tree
        self.move_tree.flip_tree()
        # Regenerating the piece images depending on the flipped state
        if config.flipped_assets:
            self.piece_images = generate_piece_images(self.flipped)
            self.update_images()
        # Flipping the board image
        self.image = pygame.transform.flip(self.image, True, False)

    def _flip_board_tiles(self):
        """
        Flips all the tiles on the board.

        This method iterates through all the tiles on the board, flipping each tile 
        and updating the board's internal state to reflect the new orientation.

        Args:
            None

        Returns:
            None
        """
        flipped_board = {}
        for pos, tile in self.board.items():
            tile.flip()
            flipped_board[flip_pos(pos)] = tile
        self.board = flipped_board

    def update_images(self):
        """
        Updates the images of all pieces on the board.

        This method iterates through all tiles on the board and updates each piece's 
        image based on the current flipped state and the piece's color and notation.

        Args:
            None

        Returns:
            None
        """
        for tile in self.board.values():
            tile.piece.update_image(self.piece_images[("w" if tile.piece.color == 1 else "b") + tile.piece.notation])

    def get_player(self, color: int) -> Player:
        """
        Retrieves the player associated with the given color.

        This method returns the current player if the provided color matches the 
        current turn, otherwise it returns the waiting player.

        Args:
            color (int): The color of the player to retrieve, where 1 typically represents white 
                and -1 represents black.

        Returns:
            Player: The player corresponding to the given color.
        """
        return self.current_player if color == self.turn else self.waiting_player

    def highlight_tile(self, highlight_color: int, *list_pos):
        """
        Highlights the specified tiles with the given color.

        This method updates the highlight color of the specified tiles. If a tile is already 
        highlighted with the given color, it is unhighlighted. It also checks the current move 
        and selected piece to apply additional highlighting logic.

        Args:
            highlight_color (int): The color to use for highlighting the tile(s).
            *list_pos (tuple[int, int]): The positions of the tiles to be highlighted.

        Returns:
            None
        """
        for pos in list_pos:
            tile = self.get_tile(pos)
            if tile.highlight_color != highlight_color:
                tile.highlight_color = highlight_color
            else:
                tile.highlight_color = None
                current_move = self.get_current_move()
                if current_move is not None:
                    to_pos = current_move.to_pos if not current_move.castling else (current_move.to_pos[0], flip_pos(castling_king_column[(1 if current_move.to_pos[1] > current_move.from_pos[1] else -1)*self.flipped], flipped=self.flipped))
                    if pos in [current_move.from_pos, to_pos]:
                        tile.highlight_color = 3
                if self.selected is not None and self.selected.piece is not None:
                    tile.highlight_color = 4

    def clear_highlights(self):
        """
        Clears all highlights from the board.

        This method removes the highlight color from all tiles on the board.

        Args:
            None

        Returns:
            None
        """
        for tile in self.board.values():
            tile.highlight_color = None

    def update_highlights(self):
        """
        Updates the highlights on the board based on the current move and selected piece.

        This method clears all existing highlights, then highlights the tiles involved in 
        the current move (including the starting and ending positions). It also highlights 
        the currently selected piece's tile.

        Args:
            None

        Returns:
            None
        """
        self.clear_highlights()
        current_move = self.get_current_move()
        if current_move is not None:
            to_pos = current_move.to_pos if not current_move.castling else (current_move.to_pos[0], flip_pos(castling_king_column[(1 if current_move.to_pos[1] > current_move.from_pos[1] else -1)*self.flipped], flipped=self.flipped))
            self.highlight_tile(3, current_move.from_pos, to_pos)
        if self.selected is not None and self.selected.piece is not None:
            self.highlight_tile(4, self.selected.pos)

    def get_current_move(self):
        """
        Retrieves the current move from the move tree.

        This method returns the move that is currently being processed in the move tree.

        Args:
            None

        Returns:
            Move: The current move being processed.
        """
        return self.move_tree.current.move
    
    def get_center(self) -> list[tuple[int, int]]:
        """
        Calculates the center positions of the board.

        This method returns a list of positions representing the center of the board. 
        The center may consist of one or more tiles depending on the board's size 
        (even or odd number of rows and columns).

        Args:
            None

        Returns:
            list[tuple[int, int]]: A list of tuples representing the center positions on the board.
        """
        mid_x, mid_y = (config.columns - 1) // 2, (config.rows - 1) // 2
        return [(mid_x + i, mid_y + j) for i in range(2 - config.columns % 2) for j in range(2 - config.rows % 2)]

    def __str__(self):
        """
        Returns a string representation of the current board state in FEN format.

        This method generates a FEN (Forsyth-Edwards Notation) string representing 
        the current state of the chessboard, including piece positions, active turn, 
        castling rights, en passant status, and move counters.

        Args:
            None

        Returns:
            str: A string representing the current board state in FEN format.
        """
        fen = []
        for row in range(config.rows):
            empty = 0
            row_fen = ""
            for col in range(config.columns):
                piece = self.get_piece((row, col))
                if piece is not None:
                    if empty > 0:
                        row_fen += str(empty)
                        empty = 0
                    row_fen += piece_to_notation(type(piece)) if piece.color == 1 else piece_to_notation(type(piece)).lower()
                else:
                    empty += 1
            if empty:
                row_fen += str(empty)
            fen.append(row_fen)
        fen = "/".join(fen)
        turn = "w" if self.turn == 1 else "b"
        castling = "".join(
            k for color in [1, -1] for k in ("KQ" if color == 1 else "kq") if self.castling[color][1 if k.upper() == "K" else -1]
        ) or "-"
        en_passant = "-"
        if self.en_passant is not None:
            d_ep = en_passant_direction[self.en_passant[0]]
            if self._is_valid_en_passant((self.en_passant[0] + d_ep, self.en_passant[1]), self.en_passant):
                en_passant = chr(97 + flip_pos(self.en_passant[1], flipped = self.flipped)) + str(flip_pos(self.en_passant[0], flipped = -self.flipped) + 1)
        return f"{fen} {turn} {castling} {en_passant} {self.half_moves} {self.full_moves}"

    def to_matrix(self):
        """
        Converts the board state to a matrix representation.

        This method creates a 14x8x8 matrix where each channel represents different 
        aspects of the game state. The first 12 channels correspond to the pieces 
        (6 for white pieces and 6 for black pieces), the 13th channel marks the 
        legal moves for the currently active player, and the 14th channel indicates 
        if the piece at a position is the currently selected piece. 

        Args:
            None

        Returns:
            np.ndarray: A 14x8x8 matrix representing the current board state.
        """
        matrix = np.zeros((14, 8, 8))
        for pos, tile in self.board.items():
            piece = tile.piece
            if piece:
                channel = piece_to_num(type(piece))
                if piece.color == -1:
                    channel += 6
                matrix[channel, pos[0], pos[1]] = 1
                if piece.color == self.turn:
                    moves = piece.calc_moves(self, pos)
                    if moves:
                        legal_moves = 0
                        for move in moves:
                            if self.convert_to_move(pos, move).is_legal(self):
                                matrix[13, move[0], move[1]] = 1 
                                legal_moves += 1
                            if legal_moves:       
                                matrix[12, pos[0], pos[1]] = 1               
        return matrix

    def convert_uci_to_move(self, uci_move):
        """
        Converts a UCI move string into a Move object.

        This method parses a UCI (Universal Chess Interface) move string, extracting 
        the starting and ending positions, and checks if the move is valid. If valid, 
        it returns the corresponding Move object, including promotion if applicable.

        Args:
            uci_move (str): The UCI move string (e.g., "e2e4" or "e7e8q").

        Returns:
            Move or None: A Move object if the UCI string corresponds to a valid move, 
            or None if the move is invalid.
        """
        columns = {"a":0, "b":1, "c":2, "d":3, "e":4, "f":5, "g":6, "h":7}
        from_pos = (8-int(uci_move[1]), columns[uci_move[0]])
        to_pos = (8-int(uci_move[3]), columns[uci_move[2]])
        promotion = notation_to_piece(uci_move[4]) if len(uci_move) == 5 else None
        if self.is_empty(from_pos):
            return None
        exist = False
        for move in self.current_player.get_moves(self):
            if (move.from_pos, move.to_pos) == (from_pos, to_pos):
                exist = True
        if not exist:
            return None
        return self.convert_to_move(from_pos, to_pos, promotion)

    def update_history(self):
        """
        Updates the history of moves displayed on the board.

        This method retrieves the list of moves from the move tree, truncates it if necessary 
        to display only the last 20 moves, and then creates a series of labels to display 
        the move history. The moves are displayed in two columns, with move notations 
        and corresponding move numbers.

        Args:
            None

        Returns:
            None
        """
        moves = self.move_tree.get_root_to_leaf()
        start_num = max(1, ceil((len(moves) - 20) / 2)) if len(moves) > 20 else 1
        moves = moves[-(22 if len(moves) % 2 == 0 else 21):]

        self.history = [
            Label(
                center=(config.width * 0.7 + (config.width * 0.1 * (i % 2)), 
                        config.height * 0.1 + (config.height * 0.035) * (i - i % 2)),
                text=move.notation,
                font_name=Fonts.TYPE_MACHINE,
                font_size=int(config.height * 0.05),
                color=Colors.WHITE.value
            )
            for i, move in enumerate(moves)
        ] + [
            Label(
                center=(config.width * 0.6, config.height * 0.1 + (config.height * 0.035) * 2 * i),
                text=f"{start_num + i}.",
                font_name=Fonts.TYPE_MACHINE,
                font_size=int(config.height * 0.05),
                color=Colors.WHITE.value
            )
            for i in range(ceil(len(moves) / 2))
        ]