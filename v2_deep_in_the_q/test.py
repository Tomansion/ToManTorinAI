import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Importing the environment
from santorinai.player import Player
from santorinai.board import Board
from santorinai.pawn import Pawn
from santorinai.tester import Tester
from santorinai.player_examples.random_player import RandomPlayer
from santorinai.player_examples.first_choice_player import FirstChoicePlayer
from santorinai.board_displayer.board_displayer import init_window, update_board


class DeepInTheQ(Player):
    def __init__(self):
        self.state_size = 225
        self.action_size1 = 8
        self.action_size2 = 8
        self.nb_fail = 0
        self.totalTurn = 0

        print("Loading model...")
        self.model = self.build_model()
        self.load("models/agent.h5")
        print("Model loaded")

    def name(self):
        return "DeepInTheQ"

    def build_model(self):
        # Define the input layer
        inputs = Input(shape=(225))

        # Shared layers
        hidden1 = Dense(64, activation="relu")(inputs)
        hidden2 = Dense(64, activation="relu")(hidden1)

        # Output layer 1 for range 1
        output1 = Dense(self.action_size1, activation="softmax")(hidden2)

        # Output layer 2 for range 2
        output2 = Dense(self.action_size2, activation="softmax")(hidden2)

        # Define the model
        model = Model(inputs=inputs, outputs=[output1, output2])

        # Compile the model
        model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(lr=0.001),
            metrics=["accuracy"],
        )

        return model

    def act(self, board: Board):
        state = board_to_state(board)
        act_values = self.model.predict(state, verbose=0)
        move_ac = np.argmax(act_values[0][0])
        build_ac = np.argmax(act_values[1][0])
        move, build = action_number_to_pos(board, move_ac, build_ac)

        return move, build

    def load(self, name):
        self.model.load_weights(name)

    def place_pawn(self, board: Board, pawn: Pawn):
        """
        Place a pawn given a board
        :param board: the board
        :param pawn: the pawn that needs to be placed
        :return: a position of the form (x, y)

        Return example: (2, 2) means that the player wants to place his pawn at the center of the board
        """
        # Placement of the pawn
        possibilites = board.get_possible_movement_positions(pawn)
        return possibilites[0]

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
        (move, build) = self.act(board)

        success, error = board.play_move(move, build)

        self.totalTurn += 1
        if not success:
            self.nb_fail += 1
            print(
                "We failed to play the move :(  "
                + str(self.nb_fail)
                + " times over "
                + str(self.totalTurn)
            )
            print(error)
            # Get a new random action :,(
            possibilites = board.get_possible_movement_and_building_positions(pawn)
            return possibilites[0]

        return move, build


def board_to_state(board: Board):
    state = np.zeros(225)
    our_pawn = board.pawn_turn
    if our_pawn == 1:
        enemy_pawn_1 = 2
        ally_pawn = 3
        enemy_pawn_2 = 4
    if our_pawn == 2:
        enemy_pawn_1 = 3
        ally_pawn = 4
        enemy_pawn_2 = 1
    if our_pawn == 3:
        enemy_pawn_1 = 4
        ally_pawn = 1
        enemy_pawn_2 = 2
    if our_pawn == 4:
        enemy_pawn_1 = 1
        ally_pawn = 2
        enemy_pawn_2 = 3

    for i in range(5):
        for j in range(5):
            state[i * 5 + j] = 1 if board.board[i][j] == 0 else 0
            state[25 + i * 5 + j] = 1 if board.board[i][j] == 1 else 0
            state[50 + i * 5 + j] = 1 if board.board[i][j] == 2 else 0
            state[75 + i * 5 + j] = 1 if board.board[i][j] == 3 else 0
            state[100 + i * 5 + j] = 1 if board.board[i][j] == 4 else 0

            state[125 + i * 5 + j] = (
                1 if board.pawns[our_pawn - 1].pos == (i, j) else 0
            )  # Our pawn
            state[150 + i * 5 + j] = (
                1 if board.pawns[ally_pawn - 1].pos == (i, j) else 0
            )  # Ally pawn
            state[175 + i * 5 + j] = (
                1 if board.pawns[enemy_pawn_1 - 1].pos == (i, j) else 0
            )  # Enemy pawn 1
            state[200 + i * 5 + j] = (
                1 if board.pawns[enemy_pawn_2 - 1].pos == (i, j) else 0
            )  # Enemy pawn 2
    return np.reshape(state, (1, 225))


def action_number_to_pos(board: Board, move, build):
    # actions = (move, build) with move and build in [0, 7] and [0, 7]
    # Pos goal: (x, y) with x and y in [0, 4]
    # Build goal: (x, y) with x and y in [0, 4]

    pawn_pos = board.get_playing_pawn().pos

    # | Vec   |      |      |     |     | Id  |     |     |
    # | ----- | ---- | ---- | --- | --- | --- | --- | --- |
    # | -1 1  | 0 1  | 1 1  |     |     | 0   | 1   | 2   |
    # | -1 0  | ---  | 1 0  |     |     | 7   | --- | 3   |
    # | -1 -1 | 0 -1 | 1 -1 |     |     | 6   | 5   | 4   |

    vec_map = {
        0: (-1, 1),
        1: (0, 1),
        2: (1, 1),
        3: (1, 0),
        4: (1, -1),
        5: (0, -1),
        6: (-1, -1),
        7: (-1, 0),
    }

    move_vec = vec_map[move]
    build_vec = vec_map[build]

    move_pos = (pawn_pos[0] + move_vec[0], pawn_pos[1] + move_vec[1])
    build_pos = (pawn_pos[0] + build_vec[0], pawn_pos[1] + build_vec[1])

    return move_pos, build_pos


tester = Tester()
tester.delay_between_moves = 0.0
tester.display_board = False
tester.verbose_level = 1

agent = DeepInTheQ()
random_player = RandomPlayer()
first_choice_player = FirstChoicePlayer()

nb_games = 10
# tester.play_1v1(agent, random_player, nb_games=nb_games)
# print("nb_no_value: ", agent.nb_no_value)
# agent.nb_no_value = 0
# tester.play_1v1(random_player, agent, nb_games=nb_games)
# print("nb_no_value: ", agent.nb_no_value)
# agent.nb_no_value = 0
# tester.play_1v1(agent, first_choice_player, nb_games=nb_games)
# agent.nb_no_value = 0
tester.play_1v1(first_choice_player, agent, nb_games=nb_games)
tester.play_1v1(agent, first_choice_player, nb_games=nb_games)
tester.play_1v1(random_player, agent, nb_games=nb_games)
tester.play_1v1(agent, random_player, nb_games=nb_games)
