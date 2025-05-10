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
                    if char.upper() == "K":
                        self.get_player(color).king = (row, column)

                    column += 1

    def _initialize_castling(self, castling_part: str):
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
        row, king_col = self.get_player(color).king
        for col in range(king_col, -1 if direction == -1 else config.columns, direction):
            piece = self.get_piece((row, col))
            if piece and piece.notation == "R" and piece.color == color:
                return True
        return False

    def _parse_en_passant(self, en_passant_part: str):
        if en_passant_part in ['-', '–']:
            return None
        if len(en_passant_part) != 2 or not en_passant_part[0].isalpha() or not en_passant_part[1].isdigit():
            raise ValueError(f"Invalid en passant notation: {en_passant_part}")

        row = flip_pos(int(en_passant_part[1]) - 1, flipped=-self.flipped)
        col = flip_pos(ord(en_passant_part[0]) - ord('a'), flipped=self.flipped)
        return row, col

    def _transform_960_fen(self, fen: str):
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
        if config.rules["king_of_the_hill"] == True and self.waiting_player.king in self.get_center():
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
        positions = [move.fen.split(" ")[:4] for move in self.move_tree.get_root_to_leaf()[self.last_irreversible_move:]]
        return any(positions.count(pos) >= 3 for pos in positions)

    def is_stalemate(self):
        return len(self.current_player.get_legal_moves(self)) == 0
    
    def is_insufficient_material(self):
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
        return sum(1 for tile in self.board.values() if tile.piece is not None)
    
    def find_tile(self, notation, color):
        return next((tile for tile in self.board.values() if tile.piece and tile.piece.notation == notation and tile.piece.color == color), None)
    
    def convert_to_move(self, from_pos, to_pos, promotion=None):
        return Move(self, from_pos, to_pos, promotion)

    def get_tile(self, pos: tuple[int, int]):
        return self.board.get(pos, None)
    
    def get_piece(self, pos: tuple[int, int]):
        try:
            return self.get_tile(pos).piece
        except AttributeError:
            raise ValueError(f"Invalid position: {pos}")
    
    def is_empty(self, pos):
        return self.get_piece(pos) is None

    def _update_castling(self, move: Move):
        piece = move.moving_piece
        if piece.notation == "K":
            # If the King moves, reset castling rights for that player
            self.castling[piece.color] = {1: False, -1: False}
        elif piece.notation == "R":
            # If the Rook moves, update the castling rights for that rook's side
            side = 1 if move.from_pos[1] > self.current_player.king[1] else -1
            self.castling[piece.color][side] = False

    def _update_last_irreversible_move(self, move: Move):
        if move.is_capture() or move.moving_piece.notation == "P" or move.castling or self.move_tree.current.castling != self.castling:
            # If the move is a capture, pawn move, castling, or a change in castling rights, mark it as irreversible
            self.last_irreversible_move = len(self.move_tree.get_root_to_leaf())

    def _is_valid_en_passant(self, pos: tuple[int, int], en_passant: tuple[int, int]):
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
        from_pos, to_pos = move.from_pos, move.to_pos
        self.en_passant = None
        if not self.is_empty(from_pos) and self.get_piece(from_pos).notation == "P" and abs(from_pos[0] - to_pos[0]) == 2:
            en_passant = ((from_pos[0] + to_pos[0]) // 2, from_pos[1])
            if self._is_valid_en_passant((to_pos[0], to_pos[1]), en_passant):
                self.en_passant = en_passant
        
    def select(self, pos: tuple[int, int]):
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
        if pos == self.selected.pos or not self.in_bounds(pos):
            self.selected = None
            return True
        return False

    def _handle_illegal_move(self, pos):
        if pos not in [move.to_pos for move in self.selected.piece.moves]:
            self.selected = None
            if self.current_player.is_king_check(self):
                play_sound(self.sounds, "illegal")
            if not self.is_empty(pos) and self.get_piece(pos).color == self.turn:
                self.select(pos)
            return True
        return False
    
    def _set_promotion(self, pos):
        if self.selected.piece.notation == "P" and pos[0] in [0, config.rows - 1]:
            self.promotion = pos
            return True
        return False

    def _select_piece(self, pos):
        if self.is_empty(pos) or self.get_piece(pos).color != self.turn:
            return
        self.selected = self.get_tile(pos)
        self.selected.piece.moves = self._filter_moves(self.selected)

    def _filter_moves(self, tile):
        moves = [self.convert_to_move(tile.pos, move) for move in tile.piece.calc_moves(self, tile.pos)]
        if config.rules["giveaway"] == True:
            if len([move for move in self.current_player.get_moves(self) if move.is_capture()]) != 0:
                return [move for move in moves if move.is_capture()]
            return [move for move in moves if not move.castling]
        else:
            return [move for move in moves if move.is_legal(self)]

    def in_bounds(self, pos: tuple[int, int]) -> bool:
        return self.get_tile(pos) is not None
    
    def flip_board(self) -> None:
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
            player.king = flip_pos(player.king)
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
        flipped_board = {}
        for pos, tile in self.board.items():
            tile.flip()
            flipped_board[flip_pos(pos)] = tile
        self.board = flipped_board

    def update_images(self):
        for tile in self.board.values():
            tile.piece.update_image(self.piece_images[("w" if tile.piece.color == 1 else "b") + tile.piece.notation])

    def get_player(self, color: int) -> Player:
        return self.current_player if color == self.turn else self.waiting_player

    def highlight_tile(self, highlight_color: int, *list_pos):
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
        for tile in self.board.values():
            tile.highlight_color = None

    def update_highlights(self):
        self.clear_highlights()
        current_move = self.get_current_move()
        if current_move is not None:
            to_pos = current_move.to_pos if not current_move.castling else (current_move.to_pos[0], flip_pos(castling_king_column[(1 if current_move.to_pos[1] > current_move.from_pos[1] else -1)*self.flipped], flipped=self.flipped))
            self.highlight_tile(3, current_move.from_pos, to_pos)
        if self.selected is not None and self.selected.piece is not None:
            self.highlight_tile(4, self.selected.pos)

    def get_current_move(self):
        return self.move_tree.current.move
    
    def get_center(self) -> list[tuple[int, int]]:
        mid_x, mid_y = (config.columns - 1) // 2, (config.rows - 1) // 2
        return [(mid_x + i, mid_y + j) for i in range(2 - config.columns % 2) for j in range(2 - config.rows % 2)]

    def __str__(self):
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