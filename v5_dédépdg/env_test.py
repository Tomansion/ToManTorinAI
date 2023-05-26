import numpy as np
from random import randint, choice

action_to_vector = {
    0: (-1, -1),
    1: (-1, 0),
    2: (-1, 1),
    3: (0, 1),
    4: (1, 1),
    5: (1, 0),
    6: (1, -1),
    7: (0, -1),
}


class Env:
    def __init__(self):
        self.board_size = 5
        self.num_actions = 8
        self.num_states = (self.board_size * self.board_size) * 3 + self.num_actions

        self.nb_win = 0
        self.reset()

    def _get_random_empty_tile(self):
        while True:
            pos = [randint(0, 4), randint(0, 4)]
            if self.board[pos[0]][pos[1]] == 0:
                return pos

    def _get_random_different_than_tile(self, value):
        while True:
            pos = [randint(0, 4), randint(0, 4)]
            if self.board[pos[0]][pos[1]] != value:
                return pos

    def _place_next_to(self, pos):
        while True:
            new_pos = [pos[0] + choice([-1, 0, 1]), pos[1] + choice([-1, 0, 1])]
            if (
                new_pos[0] >= 0
                and new_pos[0] <= 4
                and new_pos[1] >= 0
                and new_pos[1] <= 4
                and self.board[new_pos[0]][new_pos[1]] == 0
            ):
                return new_pos

    def _is_tile_accessible(self, current_pos, action):
        vector = action_to_vector[action]
        current_level = self.board[current_pos[0]][current_pos[1]]
        pos = [current_pos[0] + vector[0], current_pos[1] + vector[1]]

        # Check if the tile is in the board
        if (
            pos[0] < 0
            or pos[0] >= self.board_size
            or pos[1] < 0
            or pos[1] >= self.board_size
        ):
            return False

        # Check if we are not trying to access a higher level
        if self.board[pos[0]][pos[1]] > current_level + 1:
            return False

        return True

    def reset(self):
        self.board = [
            [0 for _ in range(self.board_size)] for _ in range(self.board_size)
        ]

        # Tower 1
        self.tower1 = [randint(0, 4), randint(0, 4)]
        self.board[self.tower1[0]][self.tower1[1]] = 1

        # Tower 2
        self.tower2 = self._place_next_to(self.tower1)
        self.board[self.tower2[0]][self.tower2[1]] = 2

        # Pawn
        if randint(0, 1) == 0:
            self.pawn_position = self._place_next_to(self.tower1)
        else:
            self.pawn_position = self._get_random_empty_tile()

        self.nb_turn = 0

        return self._get_state()

    def step(self, action):
        self.nb_turn += 1

        if not self._is_tile_accessible(self.pawn_position, action):
            if self.nb_turn >= 15:
                return self._get_state(), -1, True

            return self._get_state(), -1, False

        vector = action_to_vector[action]
        self.pawn_position = [
            self.pawn_position[0] + vector[0],
            self.pawn_position[1] + vector[1],
        ]

        if self.pawn_position == self.tower2:
            self.nb_win += 1
            return self._get_state(), 1, True

        if self.nb_turn >= 15:
            return self._get_state(), -1, True

        return self._get_state(), 0, False

    def _get_state(self):
        state = []
        # T1
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 1:
                    state.append(1)
                else:
                    state.append(0)
        # T2
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 2:
                    state.append(1)
                else:
                    state.append(0)
        # Pawn
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.pawn_position == [i, j]:
                    state.append(1)
                else:
                    state.append(0)

        # Add if we can move in each direction
        for i in range(self.num_actions):
            if self._is_tile_accessible(self.pawn_position, i):
                state.append(1)
            else:
                state.append(0)

        if len(state) != self.num_states:
            raise Exception("State size is not correct")
        return np.array(state)

    def display(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.pawn_position == [i, j]:
                    print("P", end="")
                elif self.board[i][j] == 0:
                    print("_", end="")
                else:
                    print(self.board[i][j], end="")
                print(" ", end="")
            print()

    def stats(self):
        print("Number of wins: {}".format(self.nb_win))
        print("Number of turns: {}".format(self.nb_turn))


if __name__ == "__main__":
    env = Env()
    env.reset()
    env.display()
    print(env.step(0))
    env.display()
    print(env.step(1))
