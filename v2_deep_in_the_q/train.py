import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Importing the environment
from santorinai.board import Board
from santorinai.player_examples.random_player import RandomPlayer
from santorinai.player_examples.first_choice_player import FirstChoicePlayer
from santorinai.board_displayer.board_displayer import init_window, update_board

# V2 - DeepInTheQ

# Second version of the AI. One model to move and build, input spreaded as much as possible.

# Algorithm used : [DQN](https://github.com/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb)

# ### Inputs

# - From 0 to 24 : 1 if tile is empty, 0 otherwise
# - From 25 to 49 : 1 if towel level 1
# - From 50 to 74 : 1 if towel level 2
# - From 75 to 99 : 1 if towel level 3
# - From 100 to 124 : 1 if towel terminated
# - From 125 to 149 : 1 if playing pawn
# - From 150 to 174 : 1 if ally pawn
# - From 175 to 199 : 1 if enemy pawn 1
# - From 200 to 224 : 1 if enemy pawn 2

# ### Outputs

# Movements and build vectors:

# | Vec   |      |      |     |     | Id  |     |     |
# | ----- | ---- | ---- | --- | --- | --- | --- | --- |
# | -1 1  | 0 1  | 1 1  |     |     | 0   | 1   | 2   |
# | -1 0  | ---  | 1 0  |     |     | 7   | --- | 3   |
# | -1 -1 | 0 -1 | 1 -1 |     |     | 6   | 5   | 4   |

# - From 0 to 7 : highest output level to move on
# - From 8 to 15 : highest output level to build on


class SantoriniDQNAgent:
    def __init__(self):
        self.state_size = 225
        self.action_size1 = 8
        self.action_size2 = 8
        self.memory = []
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.model = self.build_model()

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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, board: Board):
        state = board_to_state(board)

        if np.random.rand() <= self.epsilon:
            # Get random action
            possibilites = board.get_possible_movement_and_building_positions(
                board.get_playing_pawn()
            )
            (move, build) = random.choice(possibilites)

            (move_ac, build_ac) = pos_to_action_number(board, move, build)

            return state, move, build, move_ac, build_ac

        act_values = self.model.predict(state, verbose=0)
        move_ac = np.argmax(act_values[0][0])
        build_ac = np.argmax(act_values[1][0])
        move, build = action_number_to_pos(board, move_ac, build_ac)

        return state, move, build, move_ac, build_ac

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            move_ac, build_ac = action
            target_move = reward
            target_build = reward

            if not done:
                predicted_values = self.model.predict(next_state, verbose=0)
                target_move = reward + self.gamma * np.argmax(predicted_values[0][0])
                target_build = reward + self.gamma * np.argmax(predicted_values[1][0])

            predicted_values_f = self.model.predict(state, verbose=0)
            target_move_f = predicted_values_f[0][0]
            target_build_f = predicted_values_f[1][0]

            target_move_f[move_ac] = target_move
            target_build_f[build_ac] = target_build

            state = np.array([state])  # Shape: (1, 255)
            target_move_f = np.array([target_move_f])  # Shape: (8,)
            target_build_f = np.array([target_build_f])  # Shape: (8,)
            self.model.fit(
                state[0], [target_move_f, target_build_f], epochs=1, verbose=0
            )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print("epsilon", self.epsilon)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


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


def pos_to_action_number(board: Board, move_pos, build_pos):
    pawn_pos = board.get_playing_pawn().pos

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

    for key, value in vec_map.items():
        if move_pos == (pawn_pos[0] + value[0], pawn_pos[1] + value[1]):
            move = key

    for key, value in vec_map.items():
        if build_pos == (move_pos[0] + value[0], move_pos[1] + value[1]):
            build = key

    return move, build


if __name__ == "__main__":
    # Create the DQN agent
    agent = SantoriniDQNAgent()
    agent.load(f"models/agent.h5")

    player_nb = 1
    adversary_nb = 2

    random_player = RandomPlayer()

    # Iterate over episodes
    nb_episodes = 1000
    for episode in range(nb_episodes):
        board = Board(2)
        done = False

        # Iterate over time steps in the episode
        print("========= New game n°", episode)
        while not done:
            reward = 0
            # Board pawn positions
            if board.get_playing_pawn().pos == (None, None):
                # Place the pawn randomly
                # Get a random position
                possibilites = board.get_possible_movement_positions(
                    board.get_playing_pawn()
                )
                move = random.choice(possibilites)
                print("Placing pawn at ", move)
                board.place_pawn(move)
                continue

            possibilites = board.get_possible_movement_positions(
                board.get_playing_pawn()
            )
            if len(possibilites) == 0:
                # Pawn is blocked
                print("= Turn n°", board.turn_number)
                print("Pawn is blocked")
                board.next_turn()
                continue

            if board.get_playing_pawn().player_number == player_nb:
                print("= Turn n°", board.turn_number)
                # Our turn
                # Select an action
                state, move, build, move_act, build_act = agent.act(board)
                print("move:", move, "build:", build)

                # Perform the action
                move_correct, error = board.play_move(move, build)
                if not move_correct:
                    reward = -1
                    print("Invalid move, ", error)
                    # Randomly choose an action to continue the game
                    random_move, random_build = random_player.play_move(
                        board, board.get_playing_pawn()
                    )
                    board.play_move(random_move, random_build)

            else:
                # Opponent's turn
                # Randomly choose opponent's action
                board_copy = board.copy()
                (opponent_move, opponent_build) = random_player.play_move(
                    board_copy, board_copy.get_playing_pawn()
                )

                good, error = board.play_move(opponent_move, opponent_build)
                if not good:
                    print("Invalid move, ", error)
                    done = True

            # Check if the game is over
            if board.winner_player_number == adversary_nb:
                reward = -1
                print("We lost")
                done = True
            elif board.winner_player_number == player_nb:
                reward = 1
                print("We won")
                done = True
            elif board.is_game_over():
                reward = -1
                print("everyone is stuck")
                done = True

            if board.get_playing_pawn().player_number == player_nb:
                # We played, remember the state
                next_state = board_to_state(board)
                agent.remember(state, (move_act, build_act), reward, next_state, done)

                # Replay the agent's experiences to train the model
                if len(agent.memory) > 32:
                    agent.replay(32)

            if done:
                print(f"====== Episode: {episode + 1}/{nb_episodes}, Reward: {reward}")
                agent.save(f"models/agent.h5")
                break

    print("Done", episode + 1, "episodes")