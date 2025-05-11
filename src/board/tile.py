from src.config import config
from src.utils import flip_pos
from src.constants import Colors

class Tile:
    """
    Represents a single tile on the chess board, holding position, visual state, and any piece it contains.

    Attributes:
        pos (tuple[int, int]): The (row, column) coordinates of the tile on the board.
        highlight_color (Optional[Any]): Visual indicator used for UI highlighting (e.g., move suggestions or threats).
        piece (Optional[Piece]): The chess piece currently occupying the tile, if any.
    """
    def __init__(self, pos: tuple[int, int]):
        self.pos = pos
        self.calc_position()
        self.highlight_color = None
        self.piece = None

    def get_square_color(self) -> int:
        """
        Returns the color of the tile based on its board position.

        The color is determined by the parity of the sum of its row and column indices,
        which alternates between 0 (light square) and 1 (dark square) in a standard checkerboard pattern.

        Returns:
            int: 0 if the tile is a light square, 1 if it is a dark square.
        """
        return (sum(self.pos)) % 2

    def calc_moves(self, board, **kwargs) -> list[tuple[int, int]]:
        """
        Delegates the move calculation to the piece currently occupying the tile.

        This method calls the piece’s own calc_moves method using the tile’s position and
        any additional keyword arguments provided.

        Args:
            board (Board): The current game board, required for determining valid moves.
            **kwargs: Additional keyword arguments to pass through to the piece's move calculation logic.

        Raises:
            ValueError: If no piece is present on the tile when the method is called.

        Returns:
            list[tuple[int, int]]: A list of (row, column) positions representing legal or potential moves
                for the piece on this tile.
        """
        if self.piece is None:
            raise ValueError(f"No piece on the tile {self.pos}, cannot calculate moves. Board state: {str(board)}")
        return self.piece.calc_moves(board, self.pos, **kwargs)

    def calc_position(self) -> None:
        """
        Calculates the screen coordinate of the tile based on its board position and layout configuration.

        This method converts the tile's (row, column) board coordinates into pixel coordinates for rendering
        on the user interface, taking into account tile size, board margin, and the width of the evaluation bar.

        Returns:
            None: This method updates the `coord` attribute in place.
        """
        self.coord = (self.pos[1] * config.tile_size + config.margin + config.eval_bar_width, self.pos[0] * config.tile_size + config.margin)

    def flip(self) -> None:
        """
        Flips the tile's board position and updates its screen coordinate accordingly.

        This is typically used when the board orientation is reversed (e.g., flipping between players' perspectives).
        The method modifies the tile's `pos` using a position-flipping utility and recalculates its corresponding UI coordinate.

        Returns:
            None: This method updates the `pos` and `coord` attributes in place.
        """
        self.pos = flip_pos(self.pos)
        self.calc_position()

    def can_move(self, board, to: tuple[int, int]) -> bool:
        """
        Determines whether a move from this tile to the given destination is legal with respect to king safety.

        This method performs a temporary move simulation to test whether moving the piece to the target position
        would leave the player's king in check. The board state is restored after the check.

        Args:
            board (Board): The current game board, used for evaluating the legality of the move.
            to (tuple[int, int]): The destination position to test (row, column).

        Raises:
            ValueError: If no piece is present on the tile when the method is called.

        Returns:
            bool: True if the move does not place the player's king in check and is therefore legal;
                False otherwise.
        """
        if self.piece is None:
            raise ValueError(f"No piece on the tile {self.pos}, cannot move to {to}. Board state: {str(board)}")
        if self.pos == to:
            return True
        # When called, to is empty or occupied by a opponent piece
        # Save the destination square object
        save_piece = board.get_piece(to)
        self_piece = self.piece
        # Swap the piece with the destination square
        if self.piece.notation == "K":
            board.get_player(self.piece.color).king = to
        board.get_tile(to).piece = self.piece
        self.piece = None
        # Check if the king is in check after the move
        can_move = not board.current_player.is_king_check(board)
        # Restore the initial state of the board
        self.piece = self_piece
        board.get_tile(to).piece = save_piece
        if self.piece.notation == "K":
            board.get_player(self.piece.color).king = self.pos
        return can_move
    
    def get_highlight_color(self) -> tuple[int, int, int, int]:
        """
        Returns the RGBA color value used to highlight the tile based on its current highlight state.

        The color and transparency are determined by the `highlight_color` code, which corresponds
        to different user interactions or UI states:
            - 0: Right click → red
            - 1: Shift + Right click → green
            - 2: Ctrl + Right click → orange
            - 3: Move history → yellow
            - 4: Selected piece → cyan

        Returns:
            tuple[int, int, int, int]: The (R, G, B, A) color tuple used for rendering the tile's highlight.
        """
        color, a = None, None
        match self.highlight_color:
            # Right click
            case 0:
                color, a = Colors.RED.value, 75
            # Shift + Right click
            case 1:
                color, a = Colors.GREEN.value, 75
            # Ctrl + Right click
            case 2:
                color, a = Colors.ORANGE.value, 75
            # History move
            case 3:
                color, a = Colors.YELLOW.value, 75
            # Selected piece
            case 4:
                color, a = Colors.CYAN.value, 75
            # No highlight
            case _:
                color, a = Colors.TRANSPARENT.value, 0
        return *color, a