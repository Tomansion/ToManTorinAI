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

model_name = "agent_random_fighter_12000"


class ToMantoRinAI(player.Player):
    def __init__(self, model_name: str = model_name):
        super().__init__()
        self.env = Env(test=True)
        self.agent = Agent(
            self.env.get_state_size(), self.env.get_action_size(), model_name + "_best"
        )

    def name(self):
        return "ToMantoRinAI"

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
        self.env.render()
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
    from santorinai.player_examples.random_player import RandomPlayer

    # Init the tester
    tester = Tester()
    
    # tester.verbose_level = 1 # 0: no output, 1: Each game results, 2: Each move summary
    # tester.delay_between_moves = 0 # Delay between each move in seconds
    # tester.display_board = False # Display a graphical view of the board in a window
    # nb_games = 1000

    tester.verbose_level = 2 # 0: no output, 1: Each game results, 2: Each move summary
    tester.delay_between_moves = 2 # Delay between each move in seconds
    tester.display_board = True # Display a graphical view of the board in a window
    nb_games = 1

    # Init the players
    player = ToMantoRinAI()
    random_payer = RandomPlayer()

    # Play 100 games
    tester.play_1v1(player, random_payer, nb_games=nb_games)