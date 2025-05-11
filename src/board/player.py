from src.config import config
from src.board.move import Move

class Player:
    """
    Represents a player in the game with a specified color and tracking of pieces.

    Attributes:
        color (int): The color of the player, represented as an integer (e.g., 1 for white, -1 for black).
        pieces (dict): A dictionary that stores the positions of the player's pieces categorized by type
            (e.g., "P" for pawn, "R" for rook, "N" for knight, "B" for bishop, "Q" for queen, "K" for king).
        king (optional): Stores the current position of the king piece, or None if not set.
        ia (int): Represents whether the player is controlled by AI (-1 for AI, 1 for human).
    """
    def __init__(self, color: int):
        self.color = color
        # Pieces' position depending on their type
        self.pieces = {"P": [], "R": [], "N": [], "B": [], "Q": [], "K": []}
        # King's position
        self.king_pos = None
        # IA or human, -1 for Human and 1 for AI
        self.ia = -1

    def count_pieces(self) -> int:
        """
        Counts the total number of pieces the player has on the board.

        Returns:
            int: The total number of pieces of the player, including all types (pawns, rooks, knights, bishops, queens, kings).
        """
        return sum(len(pieces) for pieces in self.pieces.values())

    def add_piece(self, piece) -> None:
        """
        Adds a piece to the player's collection of pieces.

        Args:
            piece (Piece): The piece to be added to the player's collection.
                The piece will be added to the list corresponding to its notation type (e.g., "P" for pawn, "R" for rook).

        Raises:
            ValueError: If the piece's notation is not valid or not recognized by the player.

        Returns:
            None
        """
        if piece.notation not in self.pieces:
            raise ValueError(f"Invalid piece notation: {piece.notation}. Expected one of {list(self.pieces.keys())}.")
        self.pieces[piece.notation].append(piece)

    def remove_piece(self, piece) -> None:
        """
        Removes a piece from the player's collection of pieces.

        Args:
            piece (Piece): The piece to be removed from the player's collection.
                The piece will be removed from the list corresponding to its notation type (e.g., "P" for pawn, "R" for rook).
                If the piece is the king ('K'), the king attribute is also set to None.

        Raises:
            ValueError: If the piece is not found in the player's collection of pieces.

        Returns:
            None
        """
        if piece not in self.pieces[piece.notation]:
            raise ValueError(f"Piece {piece} not found in player's pieces: {self.pieces}")
        # Remove the piece from the list of pieces
        self.pieces[piece.notation].remove(piece)
        if piece.notation == 'K':
            self.king_pos = None

    def get_moves(self, board) -> list[Move]:
        """
        Generates all possible legal moves for the player’s pieces on the given board.

        Args:
            board (Board): The current game board, which contains the state of all pieces and methods for move calculations.

        Returns:
            list[Move]: A list of valid moves that the player can make, including pawn promotions when applicable.
                Each move is represented by a `Move` object, which details the starting and destination positions.
                If a pawn reaches the last row, promotion moves are generated.
        """
        moves = []
        for tile in board.board.values():
            if board.is_empty(tile.pos):
                continue
            if tile.piece.color != self.color:
                continue
            for to_pos in tile.calc_moves(board):
                if to_pos[0] in [0, config.rows - 1] and tile.piece.notation == "P":
                    for promotion in tile.piece.promotion:
                        moves.append(board.convert_to_move(tile.pos, to_pos, promotion))
                else:
                    moves.append(board.convert_to_move(tile.pos, to_pos))
        return moves
     
    def get_legal_moves(self, board) -> list[Move]:
        """
        Returns all legal moves for the piece on the current board.

        This method filters out illegal moves from the complete list of possible moves,
        retaining only those that comply with the rules of the game (e.g., no self-check).

        Args:
            board (Board): The current game board instance containing piece positions and state.

        Returns:
            list[Move]: A list of Move objects that are considered legal for this piece
                on the given board.
        """
        return [move for move in self.get_moves(board) if move.is_legal(board)]
    
    def is_king_check(self, board) -> bool:
        """
        Determines if the king of this piece's color is currently in check.

        This method checks whether the king's position is targeted by any legal move
        of the opposing player. In "giveaway" rule variants, checking the king is disabled.

        Args:
            board (Board): The current game board, containing all piece positions and logic
                for generating opponent moves.

        Returns:
            bool: True if the king is in check under standard rules; False otherwise
                or if the "giveaway" variant is active.
        """
        if config.rules["giveaway"]:
            return False
        return self.king_pos in [move.to_pos for move in board.get_player(-self.color).get_moves(board)]