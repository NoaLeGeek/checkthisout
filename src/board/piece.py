import pygame

from src.config import config
from src.utils import flip_pos
from src.constants import bishop_directions, rook_directions, queen_directions, knight_directions, castling_king_column


def notation_to_piece(notation: str) -> "Piece":
    """
    Converts a piece notation to the corresponding Piece class.

    This method takes a piece notation (e.g., "P", "K", "R", "B", "N", "Q") and 
    returns the corresponding Piece class (e.g., Pawn, King, Rook, Bishop, Knight, Queen).

    Args:
        notation (str): The notation representing the piece (e.g., "P" for Pawn, "K" for King).

    Returns:
        Piece: The class corresponding to the given notation.
    """
    return {'P':Pawn, 'K':King, 'R':Rook, 'B':Bishop, 'N':Knight, 'Q':Queen}[notation.upper()]
    
def piece_to_notation(piece: "Piece") -> str:
    """
    Converts a Piece class to its corresponding piece notation.

    This method takes a Piece class (e.g., Pawn, King, Rook, Bishop, Knight, Queen) 
    and returns the corresponding notation (e.g., "P", "K", "R", "B", "N", "Q").

    Args:
        piece (Piece): The Piece class (e.g., Pawn, King, Rook, etc.).

    Returns:
        str: The notation corresponding to the given piece class.
    """
    return {Pawn:'P', King:'K', Rook:'R', Bishop:'B', Knight:'N', Queen:'Q'}[piece]

def piece_to_num(piece) -> int:
    """
    Converts a Piece class to a corresponding numerical value.

    This method takes a Piece class (e.g., Pawn, Knight, Bishop, Rook, Queen, King) 
    and returns a numerical value that represents the piece.

    Args:
        piece (Piece): The Piece class (e.g., Pawn, Knight, Rook, etc.).

    Returns:
        int: The numerical value corresponding to the given piece class.
    """
    return {Pawn:0, Knight:1, Bishop:2, Rook:3, Queen:4, King:5}[piece]

class Piece():
    """
    Represents a chess piece.

    This class serves as the base class for all chess pieces. It stores the color 
    of the piece, its available moves, and an optional image for visual representation.

    Attributes:
        color (int): The color of the piece, where 1 represents white and -1 represents black.
        moves (list): A list of possible moves for the piece.
        image (pygame.Surface or None): An optional image for the piece's visual representation.

    Args:
        color (int): The color of the piece.
        image (pygame.Surface, optional): An optional image for the piece.
    """
    def __init__(self, color: int, image: pygame.Surface = None):
        self.color = color
        self.moves = []
        self.image = image

    def is_ally(self, piece: "Piece") -> bool:
        """
        Checks if the given piece is an ally.

        This method compares the color of the current piece with the color of another 
        piece to determine if they belong to the same player.

        Args:
            piece (Piece): The piece to compare with the current piece.

        Returns:
            bool: True if the pieces are of the same color (i.e., allies), False otherwise.
        """
        return self.color == piece.color
    
    def is_enemy(self, piece: "Piece") -> bool:
        """
        Checks if the given piece is an enemy.

        This method determines if the given piece is an enemy by checking if it is not an ally.

        Args:
            piece (Piece): The piece to compare with the current piece.

        Returns:
            bool: True if the pieces are of different colors (i.e., enemies), False otherwise.
        """
        return not self.is_ally(piece)
    
    def update_image(self, image: pygame.Surface) -> None:
        """
        Updates the image of the piece.

        This method sets the image attribute of the piece to a new provided image.

        Args:
            image (pygame.Surface): The new image to assign to the piece.

        Returns:
            None
        """
        self.image = image

class Pawn(Piece):
    def __init__(self, color: int, image: pygame.Surface = None):
        """
        Represents a Pawn piece in chess.

        This class inherits from the `Piece` class and represents a Pawn piece. In addition to the 
        basic attributes of a piece (color, moves, and image), a Pawn has a specific notation ('P') 
        and possible promotion options depending on the rules.

        Attributes:
            notation (str): The notation for the Pawn piece, set to 'P'.
            promotion (tuple): The possible promotion options for the Pawn, which depend on the game rules. 
                If giveaway is disabled, the options are Queen, Rook, Bishop, and Knight. Otherwise, 
                the Pawn can only promote to a King.

        Args:
            color (int): The color of the Pawn, where 1 represents white and -1 represents black.
            image (pygame.Surface, optional): An optional image for the Pawn piece.
        """
        super().__init__(color, image)
        self.notation = 'P'
        self.promotion = (Queen, Rook, Bishop, Knight) if config.rules["giveaway"] == False else (King)

    def calc_moves(self, board, from_pos: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Calculates the possible moves for a Pawn.

        This method computes the list of valid moves for the Pawn, considering its color, 
        the board state, and the specific rules for Pawn movement. It handles forward 
        movement, the initial two-square advance, diagonal captures, and en passant.

        Args:
            board (Board): The current game board, which provides methods for checking 
                square validity and piece status.
            from_pos (tuple[int, int]): The current position of the Pawn on the board.

        Returns:
            list[tuple[int, int]]: A list of valid destination positions the Pawn can move to.
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
    """
    Represents a Rook piece in chess.

    This class inherits from the `Piece` class and represents a Rook piece. In addition to the 
    basic attributes of a piece (color, moves, and image), a Rook has a specific notation ('R').

    Attributes:
        notation (str): The notation for the Rook piece, set to 'R'.

    Args:
        color (int): The color of the Rook, where 1 represents white and -1 represents black.
        image (pygame.Surface, optional): An optional image for the Rook piece.
    """
    def __init__(self, color: int, image: pygame.Surface = None):
        super().__init__(color, image)
        self.notation = 'R'

    def calc_moves(self, board, from_pos: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Calculates the possible moves for a Rook.

        This method computes the list of valid moves for the Rook, considering its color and the board state. 
        It iterates in all four directions (up, down, left, right) and adds valid move positions until an 
        enemy piece is encountered or the Rook reaches the edge of the board or another friendly piece.

        Args:
            board (Board): The current game board, which provides methods for checking 
                square validity and piece status.
            from_pos (tuple[int, int]): The current position of the Rook on the board.

        Returns:
            list[tuple[int, int]]: A list of valid destination positions the Rook can move to.
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
    """
    Represents a Bishop piece in chess.

    This class inherits from the `Piece` class and represents a Bishop piece. In addition to the 
    basic attributes of a piece (color, moves, and image), a Bishop has a specific notation ('B').

    Attributes:
        notation (str): The notation for the Bishop piece, set to 'B'.

    Args:
        color (int): The color of the Bishop, where 1 represents white and -1 represents black.
        image (pygame.Surface, optional): An optional image for the Bishop piece.
    """
    def __init__(self, color: int, image: pygame.Surface = None):
        super().__init__(color, image)
        self.notation = 'B'

    def calc_moves(self, board, from_pos: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Calculates the possible moves for a Bishop.

        This method computes the list of valid moves for the Bishop, considering its color and the board state. 
        It iterates in all four diagonal directions and adds valid move positions until an enemy piece is encountered 
        or the Bishop reaches the edge of the board or another friendly piece.

        Args:
            board (Board): The current game board, which provides methods for checking 
                square validity and piece status.
            from_pos (tuple[int, int]): The current position of the Bishop on the board.

        Returns:
            list[tuple[int, int]]: A list of valid destination positions the Bishop can move to.
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
    """
    Represents a Knight piece in chess.

    This class inherits from the `Piece` class and represents a Knight piece. In addition to the 
    basic attributes of a piece (color, moves, and image), a Knight has a specific notation ('N').

    Attributes:
        notation (str): The notation for the Knight piece, set to 'N'.

    Args:
        color (int): The color of the Knight, where 1 represents white and -1 represents black.
        image (pygame.Surface, optional): An optional image for the Knight piece.
    """
    def __init__(self, color: int, image: pygame.Surface = None):
        super().__init__(color, image)
        self.notation = 'N'

    def calc_moves(self, board, from_pos: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Calculates the possible moves for a Knight.

        This method computes the list of valid moves for the Knight, considering its color and the board state. 
        The Knight moves in an 'L' shape (two squares in one direction and one square perpendicular to it). 
        The method checks all possible positions the Knight can move to and adds valid positions, including 
        those that capture an enemy piece.

        Args:
            board (Board): The current game board, which provides methods for checking 
                square validity and piece status.
            from_pos (tuple[int, int]): The current position of the Knight on the board.

        Returns:
            list[tuple[int, int]]: A list of valid destination positions the Knight can move to.
        """
        self.moves = []
        for d_pos in knight_directions:
            new_pos = (from_pos[0] + d_pos[0], from_pos[1] + d_pos[1])
            if board.in_bounds(new_pos):
                if board.is_empty(new_pos) or board.get_piece(new_pos).is_enemy(self):
                    self.moves.append(new_pos)
        return self.moves


class Queen(Piece):
    """
    Represents a Queen piece in chess.

    This class inherits from the `Piece` class and represents a Queen piece. In addition to the 
    basic attributes of a piece (color, moves, and image), a Queen has a specific notation ('Q').

    Attributes:
        notation (str): The notation for the Queen piece, set to 'Q'.

    Args:
        color (int): The color of the Queen, where 1 represents white and -1 represents black.
        image (pygame.Surface, optional): An optional image for the Queen piece.
    """
    def __init__(self, color: int, image: pygame.Surface = None):
        super().__init__(color, image)
        self.notation = 'Q'

    def calc_moves(self, board, from_pos: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Calculates the possible moves for a Queen.

        This method computes the list of valid moves for the Queen, considering its color and the board state. 
        The Queen can move in straight lines (vertically and horizontally) as well as diagonally, similar to 
        a combination of a Rook and a Bishop. The method checks all possible positions the Queen can move to 
        and adds valid positions, including those that capture an enemy piece.

        Args:
            board (Board): The current game board, which provides methods for checking 
                square validity and piece status.
            from_pos (tuple[int, int]): The current position of the Queen on the board.

        Returns:
            list[tuple[int, int]]: A list of valid destination positions the Queen can move to.
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
    """
    Represents a King piece in chess.

    This class inherits from the `Piece` class and represents a King piece. In addition to the basic attributes 
    of a piece (color, moves, and image), the King has a specific notation ('K').

    Attributes:
        notation (str): The notation for the King piece, set to 'K'.

    Args:
        color (int): The color of the King, where 1 represents white and -1 represents black.
        image (pygame.Surface, optional): An optional image for the King piece.
    """
    def __init__(self, color: int, image: pygame.Surface = None):
        super().__init__(color, image)
        self.notation = 'K'

    def calc_moves(self, board, from_pos: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Calculates all the possible moves for a piece, including castling if applicable.

        Args:
            board (Board): The current game board.
            from_pos (tuple[int, int]): The starting position of the piece on the board, represented as a tuple of (row, column).

        Returns:
            list[tuple[int, int]]: A list of valid destination positions the piece can move to.
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