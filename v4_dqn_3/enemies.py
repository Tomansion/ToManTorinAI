import numpy as np
from helper import (
    new_pos_from_action,
    is_tile_accessible,
    is_outside,
)
from env import BOARD_SIZE


class random_enemy:
    # Plays anything except for invalid moves
    def get_action(self, board, our_pos, pawn_pos):
        actions = np.arange(8)
        np.random.shuffle(actions)

        new_pos = None
        for action in actions:
            new_pos_test = new_pos_from_action(our_pos, action)
            if (
                is_tile_accessible(BOARD_SIZE, board, our_pos, new_pos_test)
                and new_pos_test != pawn_pos
            ):
                new_pos = new_pos_test
                break

        if new_pos is None:
            return None, None

        np.random.shuffle(actions)

        for action in actions:
            new_build_pos = new_pos_from_action(new_pos, action)
            if (
                not is_outside(BOARD_SIZE, new_build_pos)
                and new_build_pos != pawn_pos
                and board[new_build_pos[0]][new_build_pos[1]] != 4
            ):
                return new_pos, new_build_pos

        return None, None
