# Class that uses a ToMantoRinAI model
# and respects the SantorinAI interface
from model_util import get_model
from santorinai import player
from typing import Tuple
from santorinai.pawn import Pawn
from santorinai.board import Board
import random
import json
import numpy as np

from helper import new_pos_from_action


class ToMantoRinAI(player.Player):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        print("Loading model " + model_name)
        model, env = get_model(model_name, load_enemy=False)
        self.model = model
        self.env = env

    def name(self):
        return self.model_name.split(".pth")[0]

    def place_pawn(self, board: Board, pawn: Pawn) -> Tuple[int, int]:
        """
        Place a pawn given a board
        :param board: the board
        :param pawn: the pawn that needs to be placed
        :return: a position of the form (x, y)
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
        """

        # Set the board to the env
        self.env.board = board
        # Get the state and possible actions
        state = self.env._get_obs()
        mask = self.env.action_masks()

        # Get the action choice
        action, _ = self.model.predict(state, action_masks=np.array(mask))

        # Get the move and build positions
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
        state = self.env._get_obs()
        mask = self.env.action_masks()

        # Get the action choice
        action, _ = self.model.predict(state, action_masks=np.array(mask))

        # Get the move and build positions
        move_action = int(action // 8)
        build_action = int(action % 8)
        move_pos = new_pos_from_action(pawn.pos, move_action)
        build_pos = new_pos_from_action(move_pos, build_action)

        return move_pos, build_pos


class ToMantoRinAIGuidedPlaced(ToMantoRinAIGuided):
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
        return self.model_name.split(".pth")[0] + "_placed"

    def place_pawn(self, board: Board, pawn: Pawn) -> Tuple[int, int]:
        """
        Place a pawn given a board
        :param board: the board
        :param pawn: the pawn that needs to be placed
        :return: a position of the form (x, y)
        """
        possible_positions = board.get_possible_movement_positions(pawn)
        best_spawns = [(0, 0), (1, 0), (0, 1), (1, 2), (2, 1), (1, 1)]
        for pos in best_spawns:
            if pos in possible_positions:
                return pos

        return random.choice(possible_positions)


if __name__ == "__main__":
    from santorinai.tester import Tester
    from santorinai.player_examples import (
        random_player,
        first_choice_player,
        basic_player,
    )
    from random import choice

    with open("config.json", "r") as f:
        conf = json.load(f)

    model_name = conf["model"]["name"]
    enemies = []
    for enemy in conf["enemies"]:
        if enemy == "random":
            enemies.append(random_player.RandomPlayer())
        elif enemy == "first_choice":
            enemies.append(first_choice_player.FirstChoicePlayer())
        elif enemy == "basic":
            enemies.append(basic_player.BasicPlayer())
        else:
            enemies.append(ToMantoRinAIGuided(enemy))

    # Init the tester
    tester = Tester()

    tester.verbose_level = 0  # 0: no output, 1: Each game results, 2: move
    tester.delay_between_moves = 0  # Delay between each move in seconds
    tester.display_board = False  # Display a graphical view
    nb_games = 1000

    tester.verbose_level = 2
    tester.delay_between_moves = 0.3
    tester.display_board = True
    nb_games = 1

    # Init the players
    # player = ToMantoRinAI(model_name=model_name)
    ai_player = ToMantoRinAIGuidedPlaced(model_name=model_name)

    # Play 100 games
    enemy = choice(enemies)
    print(f"Playing {nb_games} games against {enemy.name()}")
    print(f"Player: {ai_player.name()} vs Enemy: {enemy.name()}")
    res1 = tester.play_1v1(player1=ai_player, player2=enemy, nb_games=nb_games)
    print(res1)
    print(f"ai_player: {enemy.name()} vs Enemy: {ai_player.name()}")
    res2 = tester.play_1v1(player1=enemy, player2=ai_player, nb_games=nb_games)
    print(res2)

    # Get the results
    nb_win = (res1[ai_player.name()] + res2[ai_player.name()]) / 2
    print(f"{nb_win} wins out of {nb_games} games: {nb_win/nb_games*100}%")
