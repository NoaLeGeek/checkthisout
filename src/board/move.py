from src.config import config
from src.constants import castling_king_column
from src.board.piece import piece_to_notation
from src.utils import flip_pos, sign, get_value, debug_print, play_sound

class Move:
    """Represents a chess move, including special moves such as castling, en passant, and promotion.

    This class encapsulates the logic and properties of a single move on a chess board,
    including validation of standard and special moves.

    Attributes:
        from_pos (tuple[int, int]): The starting position of the move (row, column).
        to_pos (tuple[int, int]): The destination position of the move (row, column).
        moving_piece (Piece): The piece being moved.
        captured_piece (Piece or None): The piece being captured, if any.
        en_passant (bool): Indicates whether the move is an en passant capture.
        castling (bool): Indicates whether the move is a castling move.
        promotion (str or None): The notation of the piece to promote to (e.g., 'Q' for queen), or None if not a promotion.
        notation (str or None): The algebraic notation of the move, if computed.
        fen (str or None): The FEN string representing the board state after the move, if computed.

    Raises:
        ValueError: If there is no piece at from_pos.
        ValueError: If promotion is specified but the move is not a valid pawn promotion.
    """
    def __init__(self, board, from_pos, to_pos, promotion=None):
        if board.is_empty(from_pos):
            raise ValueError(f"There is no piece at {from_pos}")
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.moving_piece = board.get_piece(from_pos)
        if promotion is not None and (to_pos[0] not in [0, config.rows - 1] or self.moving_piece.notation != "P"):
            raise ValueError("Promotion is only possible for pawns at the last row")
        self.en_passant = self._is_en_passant(board)
        self.captured_piece = board.get_piece((self.from_pos[0], board.en_passant[1])) if self.en_passant else board.get_piece(to_pos) 
        self.castling = self._is_castling(board)
        self.promotion = promotion
        self.notation = None
        self.fen = None
    
    def is_capture(self) -> bool:
        """Determines whether the move is a capture.

        Returns:
            bool: True if the move results in the capture of an opponent's piece, False otherwise.
        """
        return self.captured_piece is not None

    def flip_move(self) -> None:
        """Flips the move coordinates vertically on the board.

        This is typically used to transform the move perspective from one player to the other
        by flipping both the starting and destination positions.

        Returns:
            None
        """
        self.from_pos = flip_pos(self.from_pos)
        self.to_pos = flip_pos(self.to_pos)

    def execute(self, board) -> None:
        """Executes the move on the given board, updating the game state accordingly.

        This method applies the move to the board, handling special cases such as castling, en passant,
        pawn moves, and captures. It also updates move counters, game history, FEN notation, and evaluates
        the new board state. The move is recorded in the move tree for potential undo functionality.

        Args:
            board (Board): The chess board on which the move is to be executed.

        Returns:
            None

        Raises:
            AttributeError: If required board attributes (e.g., move_tree, negamax) are not defined.
        """
        # All the things to update when the move is done for the first time
        if config.rules["giveaway"] == False:
            board._update_castling(self)
        board._update_en_passant(self)
        board._update_last_irreversible_move(self)
        board.half_moves += 1
        # Reset half_moves if it's a capture, castling or a pawn move
        if self.is_capture() or self.castling or (not board.is_empty(self.to_pos) and self.captured_piece.notation == "P"):
            board.half_moves = 0
        if board.turn == -1:
            board.full_moves += 1
        # Remember the move for undo
        board.move_tree.add(board, MoveNode(self, board))
        # This is the board state after the move
        self.fen = str(board)
        self.notation = self.to_notation(board)
        board.update_history()
        board.check_game()
        board.score = board.negamax.evaluate_board(board)

    def move(self, board):
        """Applies the piece movement to the board, including handling promotion and turn switching.

        This method updates the board state by moving the piece from its source to its destination.
        If the move involves a promotion, it performs the promotion instead. It also switches the active
        player, clears the selected piece, and updates the check counter if the '+3 checks' rule is enabled.

        Args:
            board (Board): The chess board on which the move is to be applied.

        Returns:
            None
        """
        # Update the board state
        if self.promotion is not None:
            self.promote_piece(board, self.promotion)
        else:
            self.move_piece(board)
        board.turn *= -1
        board.selected = None
        board.current_player, board.waiting_player = board.waiting_player, board.current_player
        if config.rules["+3_checks"] == True and board.current_player.is_king_check(board):
            board.checks[board.waiting_player.color] += 1

    def move_piece(self, board):
        """Performs the movement of a piece on the board, including special cases such as castling and en passant.

        This method updates the board by moving the piece to its destination and handling side effects
        such as capturing opponent pieces, updating the king's position, executing en passant captures,
        and processing castling. It also applies the '+3 checks' rule if enabled.

        Args:
            board (Board): The chess board where the piece movement is to be applied.

        Returns:
            None
        """
        # Update kings' positions
        if self.moving_piece.notation == "K":
            board.current_player.king = self.to_pos

        # Update player's pieces
        if self.is_capture() and not self.castling and not self.en_passant:
            board.waiting_player.remove_piece(self.captured_piece)

        # Capture en passant
        if self.en_passant:
            board.waiting_player.remove_piece(board.get_tile((self.from_pos[0], self.to_pos[1])).piece)
            board.get_tile((self.from_pos[0], self.to_pos[1])).piece = None

        # Handle castling logic
        if self.castling:
            self._handle_castling(board)
        # Handle normal move
        else:
            self._handle_normal_move(board)
        
        # Anarchy chess
        if config.rules["+3_checks"] == True and board.current_player.is_king_check(board):
            self.checks[board.waiting_player.color] += 1

    def _handle_castling(self, board):
        """Executes the castling move on the board.

        This method moves both the king and the rook to their castling positions. It handles both
        standard castling and Chess960-specific castling logic. The method calculates the destination
        columns dynamically and updates the board state accordingly.

        Args:
            board (Board): The chess board on which the castling move is to be performed.

        Returns:
            None
        """
        from_pos, to_pos = self.from_pos, self.to_pos
        d = sign(to_pos[1] - from_pos[1])
        # Save the pieces
        king = self.moving_piece
        rook_pos = to_pos if config.rules["chess960"] == True else (to_pos[0], (7 if d == 1 else 0))
        rook = board.get_piece(rook_pos)

        # Destinations columns
        dest_king_column = flip_pos(castling_king_column[d*board.flipped], flipped=board.flipped)
        dest_rook_column = dest_king_column - d
        
        # Castling move
        board.get_tile(from_pos).piece = None
        board.get_tile(rook_pos).piece = None
        board.get_tile((from_pos[0], dest_king_column)).piece = king
        board.get_tile((from_pos[0], dest_rook_column)).piece = rook
        
    def _handle_normal_move(self, board):
        """Executes a standard piece movement on the board.

        This method moves the piece from its source position to the destination, without
        handling any special rules (such as castling, en passant, or promotion).

        Args:
            board (Board): The chess board on which the move is to be applied.

        Returns:
            None
        """
        from_pos, to_pos = self.from_pos, self.to_pos
        board.get_tile(to_pos).piece = self.moving_piece
        board.get_tile(from_pos).piece = None

    def promote_piece(self, board, type_piece):
        """Promotes a pawn to a specified piece type on the board.

        This method replaces the pawn at the destination with a new piece of the specified type.
        It also ensures the correct image for the piece is available and updates the board state accordingly.
        
        Args:
            board (Board): The chess board on which the promotion takes place.
            type_piece (class): The class of the piece to promote the pawn to (e.g., Queen, Rook, Bishop, Knight).

        Returns:
            None

        Raises:
            ValueError: If the piece image for the promoted piece is missing.
        """
        new_piece = type_piece(self.moving_piece.color)
        if config.piece_asset != "blindfold":
            piece_image_key = f"{(('w' if new_piece.color == 1 else 'b') if config.piece_asset != "mono" else "")}{new_piece.notation}"
            if piece_image_key not in board.piece_images:
                raise ValueError(f"Missing piece image for: {piece_image_key}")
            new_piece.image = board.piece_images[piece_image_key]
        board.current_player.add_piece(new_piece)
        board.get_tile(self.to_pos).piece = new_piece
        board.get_tile(self.from_pos).piece = None
        board.promotion = None

    def undo(self, board) -> None:
        """Reverts the move on the board, undoing the effects of the previous action.

        This method undoes the move by switching the turn, reversing player roles, and handling any 
        special move effects such as pawn promotion. It also adjusts the check counter if the '+3 checks'
        rule is enabled.

        Args:
            board (Board): The chess board to revert the move on.

        Returns:
            None
        """
        board.turn *= -1
        board.selected = None
        board.current_player, board.waiting_player = board.waiting_player, board.current_player
        if config.rules["+3_checks"] == True and board.current_player.is_king_check(board):
            board.checks[board.waiting_player.color] -= 1
        if self.promotion is not None:
            self.undo_promote_piece(board)
        else:
            self.undo_move_piece(board)

    def undo_promote_piece(self, board):
        """Reverts a pawn promotion on the board.

        This method undoes the promotion of a pawn, restoring the original piece at the starting position
        and removing the promoted piece. It also restores the captured piece to its original position, if applicable.

        Args:
            board (Board): The chess board to revert the promotion on.

        Returns:
            None
        """
        board.get_tile(self.from_pos).piece = self.moving_piece
        board.current_player.remove_piece(board.get_tile(self.to_pos).piece)
        board.get_tile(self.to_pos).piece = self.captured_piece

    def undo_move_piece(self, board):
        """Reverts a standard piece movement on the board.

        This method undoes a regular piece move, restoring the board state by placing the piece
        back at its original position. If the move was an en passant capture, it restores the captured piece.
        It also handles the reversal of castling and king position updates.

        Args:
            board (Board): The chess board to revert the move on.

        Returns:
            None
        """
        # Restore the board state
        board.get_tile(self.from_pos).piece = self.moving_piece
        board.get_tile((self.to_pos[0] + self.moving_piece.color*board.flipped, self.to_pos[1]) if self.en_passant else self.to_pos).piece = self.captured_piece
        if self.en_passant:
            board.get_tile(self.to_pos).piece = None

        # Handle castling
        if self.castling:
            d = sign(self.to_pos[1] - self.from_pos[1])
            rook_pos = (self.from_pos[0], self.from_pos[1] + d)
            rook = board.get_piece(rook_pos)
            dest_rook_pos = self.to_pos if config.rules["chess960"] == True else (self.to_pos[0], (7 if d == 1 else 0))
            board.get_tile(rook_pos).piece = None
            board.get_tile(dest_rook_pos).piece = rook

        # Restore king position
        if self.moving_piece.notation == "K":
            board.current_player.king = self.from_pos

        # Restore player's pieces
        if self.is_capture() and not self.castling:
            board.waiting_player.add_piece(self.captured_piece)

    def play_sound_move(self, board) -> None:
        """Plays a sound corresponding to the type of move performed.

        This method plays different sounds depending on the move type, including castling, check, promotion,
        capture, or a regular move. It chooses the appropriate sound based on the game state and the move being made.

        Args:
            board (Board): The chess board on which the move is executed.

        Returns:
            None
        """
        if self.castling:
            play_sound(board.sounds, "castle")
        elif board.current_player.is_king_check(board):
            play_sound(board.sounds, "move-check")
        elif self.promotion is not None:
            play_sound(board.sounds, "promote")
        elif self.is_capture():
            play_sound(board.sounds, "capture")
        else:
            play_sound(board.sounds, ("move-self" if board.turn * board.flipped == 1 else "move-opponent"))
    
    def is_legal(self, board) -> bool:
        """Checks whether the move is legal according to the game's rules.

        This method checks if the move is valid based on the current game state and move type. It handles
        regular moves as well as castling, ensuring that the move follows the rules for legality. Special 
        conditions such as "giveaway" rules, castling constraints, and checks are also considered.

        Args:
            board (Board): The chess board on which the move is being checked for legality.

        Returns:
            bool: True if the move is legal, False otherwise.
        """
        if not self.castling:
            if config.rules["giveaway"] == True:
                return True
            return board.get_tile(self.from_pos).can_move(board, self.to_pos)
        # Castling
        if config.rules["giveaway"] == True or board.current_player.is_king_check(board):
            return False
        d = sign(self.to_pos[1] - self.from_pos[1])
        is_legal = True
        # -1 for O-O-O, 1 for O-O
        castling_direction = d*board.flipped
        rook_pos = self.to_pos if config.rules["chess960"] == True else (self.to_pos[0], (7 if d == 1 else 0))
        dest_rook_column = flip_pos(castling_king_column[castling_direction] - castling_direction, flipped=board.flipped) * d
        dest_king_column = flip_pos(castling_king_column[castling_direction], flipped=board.flipped) * d
        start = d * min(self.from_pos[1] * d, dest_rook_column)
        end = d * max(rook_pos[1] * d, dest_king_column)
        for next_column in range(start + castling_direction, end + castling_direction, castling_direction):
            condition = board.get_tile(self.from_pos).can_move(board, (self.from_pos[0], next_column))
            is_legal = is_legal and condition
            if not is_legal:
                break
        return is_legal
    
    def _is_castling(self, board) -> bool:
        """Checks if the move is a castling move.

        This method verifies if the move being considered is a valid castling move. It checks the conditions 
        for castling based on the piece type (king), the direction of the move, and the castling rights for 
        both players. It also handles the special case for Chess960, where additional conditions are applied.

        Args:
            board (Board): The chess board on which the move is being evaluated.

        Returns:
            bool: True if the move is a valid castling move, False otherwise.
        """
        if self.moving_piece.notation != "K":
            return False
        d = 1 if self.to_pos[1] > self.from_pos[1] else -1
        if (config.rules["chess960"] == False and abs(self.from_pos[1] - self.to_pos[1]) != 2) or (config.rules["chess960"] == True and (not self.is_capture() or board.is_empty(self.to_pos) or self.captured_piece.notation != "R" or self.moving_piece.is_enemy(self.captured_piece))):
            return False
        # O-O-O castling's right
        if d == -1 and not board.castling[self.moving_piece.color][d]:
            return False
        # O-O castling's right
        elif d == 1 and not board.castling[self.moving_piece.color][d]:
            return False
        return True
    
    def _is_en_passant(self, board) -> bool:
        """Checks if the move is an en passant capture.

        This method determines if the move being considered is an en passant capture based on the 
        current position of the pawn, the position of the en passant target, and the state of the board.

        Args:
            board (Board): The chess board on which the move is being evaluated.

        Returns:
            bool: True if the move is a valid en passant capture, False otherwise.
        """
        return (
            self.moving_piece.notation == "P" and
            board.en_passant is not None and
            self.to_pos == board.en_passant
            )
        
    def to_notation(self, board) -> str:
        """Converts the move to its algebraic notation representation.

        This method generates the standard chess algebraic notation for the current move, 
        including special cases for castling, pawn promotions, captures, checks, and checkmates. 
        It takes into account the current board state, including flipped coordinates and castling rights.

        Args:
            board (Board): The chess board to generate the notation for the move.

        Returns:
            str: The algebraic notation for the move, including check or checkmate symbols if applicable.
        """    
        string = ""
        # The move is O-O or O-O-O
        if self.castling:
            string += "O" + "-O"*(get_value(sign(self.to_pos[1] - self.from_pos[1]) * board.flipped, 1, 2))
        else:
            # Add the symbol of the piece
            if self.moving_piece.notation != "P":
                string += self.moving_piece.notation
            if self.is_capture():
                # Add the starting column if it's a pawn
                if self.moving_piece.notation == "P":
                    string += chr(flip_pos(self.from_pos[1], flipped = board.flipped) + 97)
                # Add x if it's a capture
                string += "x"
            # Add the destination's column
            string += chr(flip_pos(self.to_pos[1], flipped = board.flipped) + 97)
            # Add the destination's row
            string += str(flip_pos(self.to_pos[0], flipped = -board.flipped) + 1)
            # Add promotion
            if self.promotion is not None:
                string += "=" + piece_to_notation(self.promotion)
        # Add # if it's checkmate or + if it's a check
        if board.current_player.is_king_check(board):
            if board.is_stalemate():
                string += "#"
            else:
                string += "+"
        return string
    
class MoveNode:
    """Represents a node in the move tree, holding information about a move and its state.

    A `MoveNode` encapsulates the details of a single move, its parent node (previous move), and any 
    relevant game state information such as en passant, castling rights, half-moves, full-moves, 
    and the last irreversible move. The `MoveNode` structure is used for managing move history and 
    supporting undo/redo operations in the game.

    Attributes:
        move (Move): The move associated with this node.
        parent (MoveNode): The parent node (previous move).
        children (list): List of child nodes (subsequent moves).
        en_passant (tuple): The en passant target position, if any.
        castling (list): Castling rights for both players.
        half_moves (int): The number of half-moves (plies) in the game.
        full_moves (int): The number of full moves in the game.
        last_irreversible_move (Move): The last irreversible move made.
    """
    def __init__(self, move, board):
        self.move = move
        self.parent = None
        self.children = []
        self.en_passant = board.en_passant
        self.castling = board.castling
        self.half_moves = board.half_moves
        self.full_moves = board.full_moves
        self.last_irreversible_move = board.last_irreversible_move

class MoveTree:
    """Represents a tree structure for managing the history of moves in the game.

    The `MoveTree` is used to store the sequence of moves made during the game, allowing for 
    efficient traversal and operations like undo, redo, and move history tracking. It maintains 
    a root node and the current node representing the ongoing game state.

    Attributes:
        root (MoveNode): The root node of the move tree, representing the starting position of the game.
        current (MoveNode): The current node in the move tree, representing the state of the game after the most recent move.
    """
    def __init__(self, board):
        self.root = MoveNode(None, board)
        self.current = self.root

    def add(self, board, move_node: MoveNode):
        """Adds a new move node to the move tree.

        This method adds a new `MoveNode` as a child of the current node and updates the current 
        node to the newly added one. It also adjusts the game state by moving forward in the move tree 
        history, making the newly added move the active one.

        Args:
            board (Board): The current board state, used to update the game state.
            move_node (MoveNode): The move node to be added to the move tree.

        Returns:
            None
        """
        move_node.parent = self.current
        self.current.children.append(move_node)
        self.go_forward(board, -1)

    def go_forward(self, board, index=0):
        """Moves forward in the move tree to the specified child node.

        This method advances the current node to a child node in the move tree, performing the 
        associated move on the board and updating the board state accordingly. It also plays 
        the sound corresponding to the move and updates the board highlights and history.

        Args:
            board (Board): The current board state, used to execute the move and update the board.
            index (int, optional): The index of the child node to move to. Defaults to 0 (the first child).

        Returns:
            None
        """
        if self.current.children:
            self.current = self.current.children[index]
            self.current.move.move(board)
            self.current.move.play_sound_move(board)
            board.update_highlights()
            board.update_history()

    def go_backward(self, board):
        """Moves backward in the move tree to the parent node.

        This method moves the current node to its parent node, effectively undoing the last move 
        and reverting the board state to what it was before that move. It also plays the sound 
        corresponding to the undone move and updates the board highlights and history.

        Args:
            board (Board): The current board state, used to undo the move and restore the board.
        
        Returns:
            None
        """
        if self.current.parent:
            self.current.move.undo(board)
            self.current.move.play_sound_move(board)
            self.current = self.current.parent
            board.update_highlights()
            board.update_history()

    def go_previous(self, board):
        """Cycles to the previous sibling variation in the move tree.

        This method rotates the list of sibling nodes (alternate moves from the same position),
        brings the last one to the front, then moves backward to the parent node and forward 
        to the newly rotated first child. This allows the user to cycle backward through 
        alternative move lines (variations) at a given position.

        Args:
            board (Board): The current board state, used to undo and reapply moves.

        Returns:
            None
        """
        if self.current.parent:
            siblings = self.current.parent.children
            siblings.insert(0, siblings.pop(-1))
            self.go_backward(board)
            self.go_forward(board, 0)

    def go_next(self, board):
        """Cycles to the next sibling variation in the move tree.

        This method rotates the list of sibling nodes (alternative continuations from the same position)
        by moving the first one to the end of the list. It then undoes the current move and replays the
        new first child move, allowing the user to cycle forward through move variations.

        Args:
            board (Board): The current board state, used to undo and reapply moves.

        Returns:
            None
        """
        if self.current.parent:
            siblings = self.current.parent.children
            siblings.append(siblings.pop(0))
            self.go_backward(board)
            self.go_forward(board, 0)

    def go_root(self, board):
        """Navigates to the root of the move tree.

        This method repeatedly calls go_backward to undo moves and restore the board state
        to the beginning of the game (the root node).

        Args:
            board (Board): The current board state, used to undo each move step-by-step.

        Returns:
            None
        """
        while self.current.parent:
            self.go_backward(board)

    def go_leaf(self, board):
        """Navigates to the last descendant (leaf) node in the current move sequence.

        This method repeatedly calls go_forward to advance through all child nodes from
        the current node until reaching a node with no children, effectively applying all
        subsequent moves in the current variation.

        Args:
            board (Board): The current board state, used to apply each move step-by-step.

        Returns:
            None
        """
        while self.current.children:
            self.go_forward(board)

    def get_root_to_leaf(self):
        """Retrieves the sequence of moves from the root node to the current node.

        This method traverses the move tree backward from the current node to the root,
        collecting each move along the path. The resulting list represents the complete
        move history for the current variation.

        Returns:
            List[Move]: A list of Move objects representing the path from the root to the current node.
        """
        moves = []
        current = self.current
        while current.parent:
            moves.append(current.move)
            current = current.parent
        return moves[::-1]
    
    def flip_tree(self):
        """Flips all moves in the move tree horizontally.

        This method traverses the entire move tree starting from the root and applies
        flip_move() to each Move object. It is typically used to invert the board's
        perspective (e.g., for flipping between white and black view).

        Returns:
            None
        """
        current = self.root
        queue = [current]
        while queue:
            node = queue.pop()
            if node.move:
                node.move.flip_move()
            queue.extend(node.children)