from santorinai.board import Board

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


def board_to_state(board: Board):
    state = [0 for _ in range(225)]
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

    return state


def get_new_pos_from_action(pos: tuple, action: int):
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

    pos_vec = vec_map[action]

    new_pos = (pos[0] + pos_vec[0], pos[1] + pos_vec[1])

    return new_pos

def action_number_to_pos(board: Board, action: int):
    pawn_pos = board.get_playing_pawn().pos
    return get_new_pos_from_action(pawn_pos, action)


def action_numbers_to_move(board: Board, move, build):
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
