import pygame

from src.config import config
from src.utils import flip_pos
from src.constants import bishop_directions, rook_directions, queen_directions, knight_directions, castling_king_column


def notation_to_piece(notation: str) -> "Piece":
    """
    Converts a chess piece notation into its corresponding Piece class.

    This function takes a single-character string representing a chess piece
    (e.g., 'P' for Pawn, 'K' for King, etc.) and returns the corresponding
    class of the chess piece.

    Parameters:
        notation (str): A single-character string representing the chess piece.
                        The notation is case-insensitive.

    Returns:
        Piece: The class corresponding to the given chess piece notation.

    Raises:
        KeyError: If the provided notation does not match any valid chess piece.
    """
    return {'P':Pawn, 'K':King, 'R':Rook, 'B':Bishop, 'N':Knight, 'Q':Queen}[notation.upper()]
    
def piece_to_notation(piece: "Piece") -> str:
    """
    Converts a chess piece object to its corresponding notation symbol.

    Parameters:
        piece (Piece): The chess piece to be converted. It should be an instance
                       of a class representing a chess piece (e.g., Pawn, King, Rook, etc.).

    Returns:
        str: A single-character string representing the notation of the chess piece.
             For example:
             - 'P' for Pawn
             - 'K' for King
             - 'R' for Rook
             - 'B' for Bishop
             - 'N' for Knight
             - 'Q' for Queen

    Raises:
        KeyError: If the provided piece is not one of the recognized chess piece types.
    """
    return {Pawn:'P', King:'K', Rook:'R', Bishop:'B', Knight:'N', Queen:'Q'}[piece]

def piece_to_num(piece) -> int:
    """
    Converts a chess piece class to its corresponding numeric representation.

    Parameters:
        piece (type): The class of the chess piece to be converted. 
                      Expected values are Pawn, Knight, Bishop, Rook, Queen, or King.

    Returns:
        int: An integer representing the chess piece, where:
             - 0 corresponds to Pawn
             - 1 corresponds to Knight
             - 2 corresponds to Bishop
             - 3 corresponds to Rook
             - 4 corresponds to Queen
             - 5 corresponds to King
    """
    return {Pawn:0, Knight:1, Bishop:2, Rook:3, Queen:4, King:5}[piece]

class Piece():
    def __init__(self, color: int, image: pygame.Surface = None):
        """
        Initializes a Piece object.

        Attributes:
            color (int): The color of the piece, typically represented as an integer (e.g., 0 for white, 1 for black).
            moves (list): A list to store the possible moves for the piece. Initially empty.
            image (pygame.Surface): The graphical representation of the piece. Defaults to None if not provided.

        Args:
            color (int): The color of the piece.
            image (pygame.Surface, optional): The image representing the piece. Defaults to None.
        """
        self.color = color
        self.moves = []
        self.image = image

    def is_ally(self, piece: "Piece") -> bool:
        """
        Determines if the given piece is an ally of the current piece.

        Args:
            piece (Piece): The piece to compare with the current piece.

        Returns:
            bool: True if the given piece has the same color as the current piece, 
                  indicating they are allies; otherwise, False.
        """
        return self.color == piece.color
    
    def is_enemy(self, piece: "Piece") -> bool:
        """
        Determines if the given piece is an enemy piece.

        This function checks whether the provided piece belongs to an opposing player
        by verifying that it is not an ally.

        Parameters:
            piece (Piece): The piece to check against the current piece.

        Returns:
            bool: True if the given piece is an enemy, False otherwise.
        """
        return not self.is_ally(piece)
    
    def update_image(self, image: pygame.Surface) -> None:
        """
        Updates the image of the piece.

        Parameters:
            image (pygame.Surface): The new image to be assigned to the piece.
        """
        self.image = image

class Pawn(Piece):
    def __init__(self, color: int, image: pygame.Surface = None):
        """
        Initializes a chess piece with a specified color and optional image.

        Parameters:
            color (int): The color of the piece, typically 0 for white and 1 for black.
            image (pygame.Surface, optional): The graphical representation of the piece. Defaults to None.

        Attributes:
            notation (str): The standard chess notation for the piece. Defaults to 'P' for pawn.
            promotion (tuple or type): The possible pieces for promotion. Defaults to (Queen, Rook, Bishop, Knight) 
                                       unless the "giveaway" rule is enabled in the configuration, in which case it is (King).
        """
        super().__init__(color, image)
        self.notation = 'P'
        self.promotion = (Queen, Rook, Bishop, Knight) if config.rules["giveaway"] == False else (King)

    def calc_moves(self, board, from_pos: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Calculates all possible moves for a pawn from a given position on the board.
        This function determines the valid moves for a pawn based on its current position,
        the state of the board, and the pawn's color. It includes standard forward moves,
        the initial two-square advance, diagonal captures, and en passant captures.
        Parameters:
            board (Board): The current state of the chessboard, which provides methods
                           to check bounds, piece positions, and other game rules.
            from_pos (tuple[int, int]): The current position of the pawn as a tuple
                                        (row, column).
        Returns:
            list[tuple[int, int]]: A list of tuples representing the valid moves for the
                                   pawn from the given position.
        """
        self.moves = []
        d = self.color * board.flipped
        # Déplacement de base vers l'avant
        if board.in_bounds((from_pos[0] - d, from_pos[1])) and board.is_empty((from_pos[0] - d, from_pos[1])):
            self.moves.append((from_pos[0] - d, from_pos[1]))
            # Premier déplacement du pion (2 cases vers l'avant)
            if from_pos[0] in [1, config.rows-2] and board.in_bounds((from_pos[0] - 2*d, from_pos[1])) and board.is_empty((from_pos[0] - 2*d, from_pos[1])):
                self.moves.append((from_pos[0] - 2*d, from_pos[1]))

        # Capture diagonale et en passant
        for d_pos in [(-d, -1), (-d, 1)]:  # Diagonales
            new_pos = (from_pos[0] + d_pos[0], from_pos[1] + d_pos[1])
            if not board.in_bounds(new_pos):
                continue
            # En passant
            if board.en_passant == new_pos and not board.is_empty((from_pos[0], from_pos[1] + d_pos[1])) and board.get_piece((from_pos[0], from_pos[1] + d_pos[1])).is_enemy(self):
                self.moves.append(new_pos)
            # Capture normale 
            if board.is_empty(new_pos):
                continue
            piece = board.get_piece(new_pos)
            if piece.is_enemy(self):
                self.moves.append(new_pos)
        return self.moves


class Rook(Piece):
    def __init__(self, color: int, image: pygame.Surface = None):
        """
        Initializes a chess piece with a specified color and optional image.

        Attributes:
            color (int): The color of the chess piece, typically represented as an integer (e.g., 0 for white, 1 for black).
            image (pygame.Surface, optional): The graphical representation of the chess piece. Defaults to None.
            notation (str): The notation symbol for the chess piece. Defaults to 'R' (e.g., for a rook).

        Parameters:
            color (int): The color of the chess piece.
            image (pygame.Surface, optional): The image representing the chess piece. Defaults to None.
        """
        super().__init__(color, image)
        self.notation = 'R'

    def calc_moves(self, board, from_pos: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Calculates all possible moves for a piece on the board from a given position.

        This function determines the valid moves for a piece based on its movement
        rules (e.g., rook-like movement in this case). It considers the boundaries
        of the board, empty squares, and enemy pieces that can be captured.

        Parameters:
            board: The game board object that provides methods to check bounds,
                   piece presence, and piece ownership.
            from_pos (tuple[int, int]): The current position of the piece on the board
                                        as a tuple of (row, column).

        Returns:
            list[tuple[int, int]]: A list of tuples representing the valid positions
                                   the piece can move to.
        """
        self.moves = []
        for d_pos in rook_directions:
            new_pos = (from_pos[0] + d_pos[0], from_pos[1] + d_pos[1])
            while board.in_bounds(new_pos):
                if board.is_empty(new_pos):
                    self.moves.append(new_pos)
                elif board.get_piece(new_pos).is_enemy(self):
                    self.moves.append(new_pos)
                    break
                else:
                    break
                new_pos = (new_pos[0] + d_pos[0], new_pos[1] + d_pos[1])
        return self.moves

class Bishop(Piece):
    def __init__(self, color: int, image: pygame.Surface = None):
        """
        Initializes a chess piece with a specified color and optional image.

        Parameters:
            color (int): The color of the chess piece. Typically, 0 for white and 1 for black.
            image (pygame.Surface, optional): The graphical representation of the chess piece. Defaults to None.

        Attributes:
            notation (str): The notation representing the piece. Defaults to 'B' (e.g., for a Bishop).
        """
        super().__init__( color, image)
        self.notation = 'B'

    def calc_moves(self, board, from_pos: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Calculates all possible moves for a piece on the board from a given position.

        This function determines the valid moves for a piece based on its movement 
        rules (e.g., bishop-like movement in this case) and the current state of the board.

        Parameters:
            board: The board object representing the current state of the chess game. 
                   It provides methods to check if a position is within bounds, 
                   if a position is empty, and to retrieve a piece at a specific position.
            from_pos (tuple[int, int]): The starting position of the piece as a tuple 
                                        (row, column).

        Returns:
            list[tuple[int, int]]: A list of tuples representing the valid positions 
                                   the piece can move to.
        """
        self.moves = []
        for d_pos in bishop_directions:
            new_pos = (from_pos[0] + d_pos[0], from_pos[1] + d_pos[1])
            while board.in_bounds(new_pos):
                if board.is_empty(new_pos):
                    self.moves.append(new_pos)
                elif board.get_piece(new_pos).is_enemy(self):
                    self.moves.append(new_pos)
                    break
                else:
                    break
                new_pos = (new_pos[0] + d_pos[0], new_pos[1] + d_pos[1])
        return self.moves


class Knight(Piece):
    def __init__(self, color: int, image: pygame.Surface = None):
        """
        Initializes a chess piece with a specified color and optional image.

        Attributes:
            color (int): The color of the chess piece, typically represented as an integer (e.g., 0 for white, 1 for black).
            image (pygame.Surface, optional): The graphical representation of the chess piece. Defaults to None.
            notation (str): The notation symbol for the piece, initialized as 'N'.

        Parameters:
            color (int): The color of the chess piece.
            image (pygame.Surface, optional): The image representing the chess piece. Defaults to None.
        """
        super().__init__( color, image)
        self.notation = 'N'

    def calc_moves(self, board, from_pos: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Calculates all possible moves for a knight piece from a given position on the board.

        This method determines the valid moves for a knight based on its movement pattern 
        (L-shaped moves) and the current state of the board. It checks if the target positions 
        are within the board boundaries and whether they are either empty or occupied by an 
        enemy piece.

        Parameters:
            board: The game board object that provides methods to check bounds, 
                   piece presence, and piece ownership.
            from_pos (tuple[int, int]): The current position of the knight on the board 
                                        as a tuple of (row, column).

        Returns:
            list[tuple[int, int]]: A list of valid target positions (row, column) 
                                   that the knight can move to.
        """
        self.moves = []
        for d_pos in knight_directions:
            new_pos = (from_pos[0] + d_pos[0], from_pos[1] + d_pos[1])
            if board.in_bounds(new_pos):
                if board.is_empty(new_pos) or board.get_piece(new_pos).is_enemy(self):
                    self.moves.append(new_pos)
        return self.moves


class Queen(Piece):
    def __init__(self, color: int, image: pygame.Surface = None):
        """
        Initializes a chess piece with a specified color and optional image.

        Attributes:
            notation (str): The notation representing the piece, default is 'Q' (Queen).

        Parameters:
            color (int): The color of the piece, typically 0 for white and 1 for black.
            image (pygame.Surface, optional): The graphical representation of the piece. Defaults to None.
        """
        super().__init__(color, image)
        self.notation = 'Q'

    def calc_moves(self, board, from_pos: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Calculates all possible moves for a piece on the board from a given position.

        This function determines the valid moves for a piece based on its movement
        directions (e.g., queen directions) and the state of the board. It considers
        whether the destination squares are empty, occupied by an enemy piece, or
        occupied by an allied piece.

        Parameters:
            board: The game board object that provides methods to check bounds,
                   piece positions, and piece ownership.
            from_pos (tuple[int, int]): The current position of the piece on the board,
                                        represented as a tuple of (row, column).

        Returns:
            list[tuple[int, int]]: A list of valid moves represented as tuples of
                                   (row, column) positions.
        """
        self.moves = []
        for d_pos in queen_directions:
            new_pos = (from_pos[0] + d_pos[0], from_pos[1] + d_pos[1])
            while board.in_bounds(new_pos):
                if board.is_empty(new_pos):  # Case non occupée
                    self.moves.append(new_pos)
                elif board.get_piece(new_pos).is_enemy(self):  # Pièce ennemie
                    self.moves.append(new_pos)
                    break
                else:  # Pièce alliée
                    break
                new_pos = (new_pos[0] + d_pos[0], new_pos[1] + d_pos[1])
        return self.moves

    
class King(Piece):
    def __init__(self, color: int, image: pygame.Surface = None):
        """
        Initializes a Piece object with a specified color and optional image.

        Parameters:
            color (int): The color of the piece, typically represented as an integer (e.g., 0 for white, 1 for black).
            image (pygame.Surface, optional): A pygame Surface object representing the visual representation of the piece. Defaults to None.

        Attributes:
            notation (str): The notation symbol for the piece, initialized as 'K'.
        """
        super().__init__(color, image)
        self.notation = 'K'

    def calc_moves(self, board, from_pos: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Calculates all possible moves for a chess piece from a given position on the board.

        This function determines the valid moves for a piece based on its movement rules, 
        the current state of the board, and special rules such as castling. It updates the 
        `self.moves` attribute with the list of valid moves.

        Parameters:
            board (Board): The current state of the chessboard, which provides methods 
                           to check bounds, piece positions, and other game rules.
            from_pos (tuple[int, int]): The current position of the piece as a tuple 
                                        (row, column).

        Returns:
            list[tuple[int, int]]: A list of tuples representing the valid positions 
                                   the piece can move to.
        """
        self.moves = []
        for d_pos in queen_directions:
            new_pos = (from_pos[0] + d_pos[0], from_pos[1] + d_pos[1])
            if board.in_bounds(new_pos) and (board.is_empty(new_pos) or board.get_piece(new_pos).is_enemy(self)):
                self.moves.append(new_pos)
        # Castling
        rooks = {1: None, -1: None}
        # -1 = O-O-O, 1 = O-O
        # Calculate possible castling
        possible_castling = []
        if board.castling[self.color][1]:
            possible_castling.append(1)
        if board.castling[self.color][-1]:
            possible_castling.append(-1)
        # Find the rook(s) that can castle
        for d in possible_castling:
            # -1 = O-O-O, 1 = O-O
            castling_direction = d*board.flipped
            for i in range(from_pos[1] + d, flip_pos(7, flipped=d) + d, d):
                # Skip if empty square
                if board.is_empty((from_pos[0], i)):
                    continue
                if rooks[castling_direction] is not None:
                    rooks[castling_direction] = None
                    possible_castling.remove(d)
                    break
                piece = board.get_piece((from_pos[0], i))
                if piece.notation == "R" and piece.is_ally(self):
                    rooks[castling_direction] = i
        # Check if the squares between the king and the found rook(s) are empty
        for d in possible_castling:
            castling_direction = d*board.flipped
            if rooks[castling_direction] is None:
                continue
            rook_column = rooks[castling_direction] * d
            dest_rook_column = flip_pos(castling_king_column[castling_direction] - castling_direction, flipped=board.flipped) * d
            dest_king_column = flip_pos(castling_king_column[castling_direction], flipped=board.flipped) * d
            start = d * min(from_pos[1] * d, dest_rook_column)
            end = d * max(rook_column, dest_king_column)
            columns = list(range(start, end + d, d))
            if all(board.is_empty((from_pos[0], i)) or i in [rooks[castling_direction], from_pos[1]] for i in columns):
                castling_column = rooks[castling_direction] if config.rules["chess960"] == True else flip_pos(castling_king_column[castling_direction], flipped=board.flipped)
                self.moves.append((from_pos[0], castling_column))
        return self.moves
