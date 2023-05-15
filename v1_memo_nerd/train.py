## V1 - DeathCursor
# Q value learning

# Importing libraries
from time import sleep
import numpy as np
import ujson as json
import os

# Importing the environment
from santorinai.board import Board
from santorinai.player_examples.random_player import RandomPlayer
from santorinai.player_examples.first_choice_player import FirstChoicePlayer
from santorinai.board_displayer.board_displayer import init_window, update_board


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
    # print(boardHash)

    return boardHash


def getVector(pos1, pos2):
    return (pos2[0] - pos1[0], pos2[1] - pos1[1])


class PlayerTraining:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value

    def chooseAction(self, board: Board):
        our_pawn = board.get_playing_pawn()
        possible_moves = board.get_possible_movement_and_building_positions(our_pawn)

        # print("possible moves:", len(possible_moves))

        if len(possible_moves) == 0:
            return (None, None)

        # take random action
        idx = np.random.choice(len(possible_moves))
        action = possible_moves[idx]

        if np.random.uniform(0, 1) > self.exp_rate:
            value_max = -999
            for move in possible_moves:
                next_board = board.copy()
                if our_pawn.pos == (None, None):
                    move_valid, error = next_board.place_pawn(move[0])
                else:
                    move_valid, error = next_board.play_move(move[0], move[1])

                if not move_valid:
                    print("Invalid move", move, error)
                    exit(1)

                next_boardHash = getBoardHash(next_board)
                value = (
                    0
                    if self.states_value.get(next_boardHash) is None
                    else self.states_value.get(next_boardHash)
                )
                # print("value", value)
                if value >= value_max:
                    value_max = value
                    action = move

        # print("We have taken the action", action)
        return action

    # append a hash state
    def addState(self, state):
        self.states.append(state)

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (
                self.decay_gamma * reward - self.states_value[st]
            )
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
        fw = open("policy_" + str(self.name) + ".json", "w")
        json.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self):
        file = "policy_" + str(self.name) + ".json"
        if not os.path.isfile(file):
            return

        fr = open(file, "r")
        self.states_value = json.load(fr)
        fr.close()


def play(player: PlayerTraining, player2, display_board, statistics):
    board = Board(2)
    if display_board:
        window = init_window(["Training in progress..."])
    # update_board(window, board)

    while not board.is_game_over():
        pawn = board.get_playing_pawn()
        if pawn.player_number == 1:
            # player 1
            action = player.chooseAction(board)
            if pawn.pos == (None, None):
                # Placing pawn
                board.place_pawn(action[0])

            else:
                # Playing move
                board.play_move(action[0], action[1])

            if display_board:
                update_board(window, board)
                sleep(0.5)

            # based on the action, get the next state S(t+1)
            boardHash = getBoardHash(board)
            player.addState(boardHash)

        else:
            # player 2, random player
            if pawn.pos == (None, None):
                # Placing pawn
                position = player2.place_pawn(board, pawn)
                board.place_pawn(position)

            else:
                board_copy = board.copy()
                (position, build) = player2.play_move(
                    board_copy, board_copy.get_playing_pawn()
                )
                board.play_move(position, build)
                # print(ok, msg)

            if display_board:
                update_board(window, board)
                sleep(0.5)

    if board.winner_player_number == 1:
        # print("Player 1 wins!")
        statistics["nb_games_won"] += 1
        player.feedReward(1)
    elif board.winner_player_number == 2:
        # print("Player 2 wins!")
        statistics["nb_games_lost"] += 1
        player.feedReward(-1)
    else:
        print("Draw!")
        statistics["nb_games_draw"] += 1
        player.feedReward(0)

    player.reset()
    statistics["nb_games"] += 1


if __name__ == "__main__":
    player = PlayerTraining("DeathCursor")
    player.loadPolicy()

    display_board = False
    statistics = {
        "nb_games": 0,
        "nb_games_won": 0,
        "nb_games_lost": 0,
        "nb_games_draw": 0,
    }

    nb_batch = 10000
    games_per_batch = 100

    for batch_nb in range(nb_batch):
        print("Training progress:", batch_nb, "/", nb_batch)

        statistics = {
            "nb_games_won": 0,
            "nb_games_lost": 0,
            "nb_games_draw": 0,
            "nb_games": 0,
        }

        # Game against random player
        random_player = RandomPlayer()
        for game_nb in range(games_per_batch):
            play(player, random_player, display_board, statistics)

        print(
            "nb games won against random player:",
            statistics["nb_games_won"],
            "(",
            statistics["nb_games_won"] / statistics["nb_games"] * 100,
            "%)",
        )
        statistics = {
            "nb_games_won": 0,
            "nb_games_lost": 0,
            "nb_games_draw": 0,
            "nb_games": 0,
        }

        # Game against first choice player
        first_choice_player = FirstChoicePlayer()
        for game_nb in range(games_per_batch):
            play(player, first_choice_player, display_board, statistics)

        print(
            "nb games won against First choice:",
            statistics["nb_games_won"],
            "(",
            statistics["nb_games_won"] / statistics["nb_games"] * 100,
            "%)",
        )
        print("nb states:", len(player.states_value))
        player.savePolicy()
