# Class that uses a ToMantoRinAI model
# and respects the SantorinAI interface
from santorinai import player
from typing import Tuple
from santorinai.pawn import Pawn
from santorinai.board import Board
from agent import Agent
import random
from env import Env
from helper import new_pos_from_action
import numpy as np


class ToMantoRinAI(player.Player):
    def __init__(self, model_name):
        super().__init__()
        self.env = Env(test=True)

        self.model_name = model_name
        self.agent = Agent(
            self.env.get_state_size(), self.env.get_action_size(), self.model_name
        )
        self.agent.load(fail_if_not_found=True)

    def name(self):
        return self.model_name.split(".pth")[0]

    def place_pawn(self, board: Board, pawn: Pawn) -> Tuple[int, int]:
        """
        Place a pawn given a board
        :param board: the board
        :param pawn: the pawn that needs to be placed
        :return: a position of the form (x, y)

        Return example: (2, 2) means that the player wants to place his pawn at the center of the board
        """
        possible_positions = board.get_possible_movement_positions(pawn)
        return random.choice(possible_positions)

    def play_move(
        self, board: Board, pawn: Pawn
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
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

        # Set the board to the env
        self.env.board = board
        # Get the state and possible actions
        state, possible_actions = self.env.get_state()

        # Get the action choice
        actions_choice = self.agent.get_action(state, possible_actions)

        # Get the move and build positions
        action = np.argmax(actions_choice)
        move_action = int(action // 8)
        build_action = int(action % 8)
        move_pos = new_pos_from_action(pawn.pos, move_action)
        build_pos = new_pos_from_action(move_pos, build_action)

        return move_pos, build_pos


class ToMantoRinAIGuided(ToMantoRinAI):
    """
    A player that answer to simple rules + use the model to play:
    - Place pawns randomly
    - If there is a winning move, play it.
    - If we can prevent the opponent from winning, do it.
    - Listen to the model
    """

    def __init__(self, model_name):
        super().__init__(model_name)

    def name(self):
        return self.model_name.split(".pth")[0] + "_guided"

    def get_ally_pawn(self, board, our_pawn):
        for pawn in board.pawns:
            if (
                pawn.number != our_pawn.number
                and pawn.player_number == our_pawn.player_number
            ):
                return pawn

    def get_enemy_pawns(self, board, our_pawn):
        pawns = []
        for pawn in board.pawns:
            if pawn.player_number != our_pawn.player_number:
                pawns.append(pawn)
        return pawns

    def get_winning_moves(self, board: Board, pawn):
        available_positions = board.get_possible_movement_positions(pawn)
        winning_moves = []
        for pos in available_positions:
            if board.board[pos[0]][pos[1]] == 3:
                winning_moves.append(pos)

        return winning_moves

    def play_move(self, board, pawn):
        available_positions = board.get_possible_movement_positions(pawn)
        board_copy = board.copy()

        # Check if we can win
        for pos in available_positions:
            if board.board[pos[0]][pos[1]] == 3:
                # We can win!
                return pos, (None, None)

        # Check if we can prevent the opponent from winning
        enemy_pawns = self.get_enemy_pawns(board, pawn)
        for enemy_pawn in enemy_pawns:
            winning_moves = self.get_winning_moves(board, enemy_pawn)
            for winning_move in winning_moves:
                for available_pos in available_positions:
                    if board.is_position_adjacent(winning_move, available_pos):
                        # We can prevent the opponent from winning
                        # Building on the winning move
                        return available_pos, winning_move

        # Listen to the model
        # Set the board to the env
        self.env.board = board_copy
        # Get the state and possible actions
        state, possible_actions = self.env.get_state()

        # Get the action choice
        actions_choice = self.agent.get_action(state, possible_actions)

        # Get the move and build positions
        action = np.argmax(actions_choice)
        move_action = int(action // 8)
        build_action = int(action % 8)
        move_pos = new_pos_from_action(pawn.pos, move_action)
        build_pos = new_pos_from_action(move_pos, build_action)

        return move_pos, build_pos


if __name__ == "__main__":
    from santorinai.tester import Tester
    from santorinai.player_examples import (
        random_player,
        first_choice_player,
        basic_player,
    )
    from random import choice

    import json

    with open("config.json", "r") as f:
        conf = json.load(f)

    model_name = conf["model"]["name"]
    enemies = []
    if conf["enemy"]["random"]:
        enemies.append(random_player.RandomPlayer())
    if conf["enemy"]["first_choice"]:
        enemies.append(first_choice_player.FirstChoicePlayer())
    if conf["enemy"]["basic"]:
        enemies.append(basic_player.BasicPlayer())

    # Init the tester
    tester = Tester()

    # tester.verbose_level = 1 # 0: no output, 1: Each game results, 2: Each move summary
    # tester.delay_between_moves = 0 # Delay between each move in seconds
    # tester.display_board = False # Display a graphical view of the board in a window
    # nb_games = 1000

    tester.verbose_level = 2  # 0: no output, 1: Each game results, 2: Each move summary
    tester.delay_between_moves = 1  # Delay between each move in seconds
    tester.display_board = True  # Display a graphical view of the board in a window
    nb_games = 1

    # Init the players

    player = ToMantoRinAIGuided(model_name=model_name)

    # Play 100 games
    tester.play_1v1(player, choice(enemies), nb_games=nb_games)
