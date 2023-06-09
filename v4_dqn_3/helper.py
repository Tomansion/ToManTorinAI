import matplotlib.pyplot as plt
import matplotlib
from random import randint, choice
import numpy as np

matplotlib.use("GTK3Agg")


def plot(mean_scores):
    plt.figure(2)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Number of games")
    plt.ylabel("Scores")
    # Set min and max values for the axes
    plt.ylim(0, 1)
    plt.plot(mean_scores)
    plt.pause(0.001)


def plot_test(plot_x, mean_scores_test, model_name="agent"):
    plt.figure(3)
    plt.clf()
    plt.title("Testing...")
    plt.xlabel("Number of games")
    plt.ylabel("Scores")
    # Set min and max values for the axes
    plt.ylim(0, 1)
    plt.plot(plot_x, mean_scores_test)
    # Save the figure
    plt.savefig(model_name + "_test_results.png")
    plt.pause(0.001)


VEC_MAP = {
    0: (-1, 1),
    1: (0, 1),
    2: (1, 1),
    3: (1, 0),
    4: (1, -1),
    5: (0, -1),
    6: (-1, -1),
    7: (-1, 0),
}


def new_pos_from_action(pos: tuple, action: int):
    # | Vec   |      |      |     |     | Id  |     |     |
    # | ----- | ---- | ---- | --- | --- | --- | --- | --- |
    # | -1 1  | 0 1  | 1 1  |     |     | 0   | 1   | 2   |
    # | -1 0  | ---  | 1 0  |     |     | 7   | --- | 3   |
    # | -1 -1 | 0 -1 | 1 -1 |     |     | 6   | 5   | 4   |

    pos_vec = VEC_MAP[action]
    new_pos = (pos[0] + pos_vec[0], pos[1] + pos_vec[1])

    return new_pos


def action_from_pos(pos: tuple, new_pos: tuple):
    for action, vec in VEC_MAP.items():
        if (pos[0] + vec[0], pos[1] + vec[1]) == new_pos:
            return action

    return None


def create_empty_board(size):
    return np.zeros((size, size), dtype=int)


def get_random_empty_tile(board):
    while True:
        pos = [randint(0, len(board) - 1), randint(0, len(board) - 1)]
        if board[pos[0]][pos[1]] == 0:
            return pos


def get_random_different_than_tile(board, value):
    while True:
        pos = [randint(0, len(board) - 1), randint(0, len(board) - 1)]
        if board[pos[0]][pos[1]] != value:
            return pos


def place_next_to(board, pos):
    while True:
        new_pos = [pos[0] + choice([-1, 0, 1]), pos[1] + choice([-1, 0, 1])]
        if (
            new_pos[0] >= 0
            and new_pos[0] <= len(board) - 1
            and new_pos[1] >= 0
            and new_pos[1] <= len(board) - 1
            and board[new_pos[0]][new_pos[1]] == 0
        ):
            return new_pos


def is_outside(BOARD_SIZE, coord):
    return (
        coord[0] > BOARD_SIZE - 1
        or coord[0] < 0
        or coord[1] > BOARD_SIZE - 1
        or coord[1] < 0
    )


def is_tile_accessible(BOARD_SIZE, board, current_pos, next_pos):
    if is_outside(BOARD_SIZE, next_pos):
        return False

    current_level = board[current_pos[0]][current_pos[1]]
    target_level = board[next_pos[0]][next_pos[1]]

    if target_level == 4:
        return False

    # Check if we are not trying to access a higher level
    return board[next_pos[0]][next_pos[1]] <= current_level + 1
