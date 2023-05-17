from santorinai.board import Board
from santorinai.player_examples.random_player import RandomPlayer
from utils import board_to_state, action_number_to_pos
from random import choice, randint


class Env:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = Board(2)
        self.random_player = RandomPlayer()
        self.player_number = randint(1, 2)
        self.opponent_player_number = 2 if self.player_number == 1 else 1
        # print("===== New game started!")
        # print("Our player number:", self.player_number)

        # Place the pawns randomly
        for i in range(4):
            playing_pawn = self.board.get_playing_pawn()
            positions = self.board.get_possible_movement_positions(playing_pawn)
            self.board.place_pawn(choice(positions))

        # Play random player if it is n°1
        if self.opponent_player_number == 1:
            self.opponent_step()

        return self.get_state()

    def get_state(self):
        return board_to_state(self.board)

    def step(self, action):
        # Play our player
        playing_pawn = self.board.get_playing_pawn()
        # print("== Our player:", playing_pawn.player_number)
        if self.player_number != self.board.get_playing_pawn().player_number:
            raise Exception("Not our turn")
        nb_possible_moves = len(
            self.board.get_possible_movement_positions(playing_pawn)
        )
        if nb_possible_moves == 0:
            # print("No possible moves")
            self.board.next_turn()
            self.opponent_step()
            return self.get_state(), -3, False

        # print("Our action:", action)
        move_pos = action_number_to_pos(self.board, action)
        board_copy = self.board.copy()
        playing_pawn_copy = board_copy.get_playing_pawn()
        playing_pawn_copy.move(move_pos)
        # Get random build position
        build_pos = board_copy.get_possible_building_positions(playing_pawn_copy)
        if len(build_pos) == 0:
            random_build_pos = (None, None)
        else:
            random_build_pos = choice(build_pos)
        move_ok, error = self.board.play_move(move_pos, random_build_pos)

        if not move_ok:
            # print(self.board.get_playing_pawn())
            # print(action, move_pos, random_build_pos)
            # print(self.board)
            # print("Wrong move:", error)
            return self.get_state(), -5, True

        if self.board.winner_player_number == self.player_number:
            # print("We won!")
            return self.get_state(), 1, True
        elif self.board.is_game_over():
            # print("Draw!")
            return self.get_state(), 0, True

        return self.opponent_step()

    def opponent_step(self):
        # Play random player
        # print("== Opponent player:", self.board.get_playing_pawn().player_number)
        playing_pawn = self.board.get_playing_pawn()
        nb_possible_moves = len(
            self.board.get_possible_movement_positions(playing_pawn)
        )
        if nb_possible_moves == 0:
            # print("No possible moves for the opponent")
            self.board.next_turn()
            return self.get_state(), 3, False

        try:
            board_copy = self.board.copy()
            move, build = self.random_player.play_move(
                board_copy, board_copy.get_playing_pawn()
            )
            move_valid, error = self.board.play_move(move, build)
            if not move_valid:
                raise Exception("Wrong move from random player: " + error)

            if self.board.winner_player_number == self.opponent_player_number:
                # print("We lost!")
                return self.get_state(), -1, True
            elif self.board.is_game_over():
                # print("Draw!")
                return self.get_state(), 0, True

            # print("Opponent played:", move, build)
            # print("== next player:", self.board.get_playing_pawn().player_number)

            return self.get_state(), 1, False
        except Exception as e:
            # print("Error while playing random player:", e)
            # print("Board:")
            # print(self.board.get_playing_pawn())
            # print(self.board)
            raise e
