from santorinai.player import Player
from santorinai.board import Board
from santorinai.pawn import Pawn
from santorinai.tester import Tester
from santorinai.player_examples.random_player import RandomPlayer
from santorinai.player_examples.first_choice_player import FirstChoicePlayer
from santorinai.board_displayer.board_displayer import init_window, update_board

import ujson as json
import os
from random import choice


def getBoardHash(board: Board):
    # get unique hash of current board state
    # Board
    board_hash = ""
    for row in board.board:
        for cell in row:
            board_hash += str(cell)

    # Pawns
    pawns = board.pawns
    pawns_hash = ""
    for pawn in pawns:
        if pawn.pos == (None, None):
            pawns_hash += "55"
        else:
            pawns_hash += str(pawn.pos[0]) + str(pawn.pos[1])

    # Pawn to play
    pawn_number = str(board.get_playing_pawn().number)

    boardHash = board_hash + pawns_hash + pawn_number

    return boardHash


class DeathCursor(Player):
    def __init__(self):
        # Load policy
        print("DeathCursor is loading policy...")
        policy_file = "policy_DeathCursor.json"
        if not os.path.isfile(policy_file):
            raise FileNotFoundError("DeathCursor policy not found")

        fr = open(policy_file, "r")
        self.states_value = json.load(fr)
        self.nb_no_value = 0
        self.tt_turn = 0
        print("DeathCursor policy loaded")

    def name(self):
        return "DeathCursor"

    def get_action(self, board: Board):
        our_pawn = board.get_playing_pawn()
        possible_moves = board.get_possible_movement_and_building_positions(our_pawn)

        if len(possible_moves) == 0:
            return (None, None)

        # Take the action with the highest value
        action = choice(possible_moves)
        value_max = -10000000
        for move in possible_moves:
            next_board = board.copy()
            if our_pawn.pos == (None, None):
                next_board.place_pawn(move[0])
            else:
                next_board.play_move(move[0], move[1])

            next_boardHash = getBoardHash(next_board)
            value = (
                value_max
                if self.states_value.get(next_boardHash) is None
                else self.states_value.get(next_boardHash)
            )
            if value > value_max:
                # print("Value of action", move, ":", value)
                # print("New value max:", value)
                value_max = value
                action = move

        self.tt_turn += 1
        if value_max == -10000000:
            # print("No value found, taking random action :,(")
            self.nb_no_value += 1
            # print(self.nb_no_value / self.tt_turn * 100, "% of no value")

        return action

    def place_pawn(self, board: Board, pawn: Pawn):
        """
        Place a pawn given a board
        :param board: the board
        :param pawn: the pawn that needs to be placed
        :return: a position of the form (x, y)

        Return example: (2, 2) means that the player wants to place his pawn at the center of the board
        """
        return self.get_action(board)[0]

    def play_move(self, board: Board, pawn: Pawn):
        """
        Play a move given a board
        :param board: the board
        :param pawn: the pawn that needs to be moved and that needs to build
        :return: two positions of the form (x1, y1), (x2, y2)

        The first coordinate corresponds to the new position of the pawn
        The second coordinate corresponds to the position of the construction of the tower

        Return example: (2, 2), (2, 3) means that the player wants to move the pawn at
        at center of the board and build a tower at the top of his position
        """
        return self.get_action(board)


tester = Tester()
tester.delay_between_moves = 0.0
tester.display_board = False
tester.verbose_level = 0

death_cursor = DeathCursor()
random_player = RandomPlayer()
first_choice_player = FirstChoicePlayer()

nb_games = 100
# tester.play_1v1(death_cursor, random_player, nb_games=nb_games)
# print("nb_no_value: ", death_cursor.nb_no_value)
# death_cursor.nb_no_value = 0
# tester.play_1v1(random_player, death_cursor, nb_games=nb_games)
# print("nb_no_value: ", death_cursor.nb_no_value)
# death_cursor.nb_no_value = 0
# tester.play_1v1(death_cursor, first_choice_player, nb_games=nb_games)
# death_cursor.nb_no_value = 0
tester.play_1v1(first_choice_player, death_cursor, nb_games=nb_games)
print(death_cursor.nb_no_value / death_cursor.tt_turn * 100, "% of no value")
death_cursor = DeathCursor()
tester.play_1v1(death_cursor, first_choice_player, nb_games=nb_games)
print(death_cursor.nb_no_value / death_cursor.tt_turn * 100, "% of no value")
death_cursor = DeathCursor()
tester.play_1v1(random_player, death_cursor, nb_games=nb_games)
print(death_cursor.nb_no_value / death_cursor.tt_turn * 100, "% of no value")
death_cursor = DeathCursor()
tester.play_1v1(death_cursor, random_player, nb_games=nb_games)
print(death_cursor.nb_no_value / death_cursor.tt_turn * 100, "% of no value")
