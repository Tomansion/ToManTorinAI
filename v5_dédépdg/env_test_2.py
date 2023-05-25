from random import choice, randint
from utils import get_new_pos_from_action

# Test env

# 5x5 board
# A lvl1,2 & 3 tower are placed randomly on the board next to each other
# goal: Go on the lvl 3 tower

# Inputs:
# - 5x5 board
# - Pawn position as a 5x5 board

num_states = 25 * 2
num_actions = 8


class Env:
    def __init__(self):
        self.reset()

    def _get_random_empty_tile(self):
        while True:
            pos = (randint(0, 4), randint(0, 4))
            if self.board[pos[0]][pos[1]] == 0:
                return pos

    def _get_random_different_than_tile(self, value):
        while True:
            pos = (randint(0, 4), randint(0, 4))
            if self.board[pos[0]][pos[1]] != value:
                return pos

    def _place_next_to(self, pos):
        while True:
            new_pos = (pos[0] + choice([-1, 0, 1]), pos[1] + choice([-1, 0, 1]))
            if (
                new_pos[0] >= 0
                and new_pos[0] <= 4
                and new_pos[1] >= 0
                and new_pos[1] <= 4
                and self.board[new_pos[0]][new_pos[1]] == 0
            ):
                return new_pos

    def reset(self):
        # Init board
        self.board = [[0 for _ in range(5)] for _ in range(5)]

        # Place the towers
        lvl1_pos = (randint(0, 4), randint(0, 4))
        self.board[lvl1_pos[0]][lvl1_pos[1]] = 1
        lvl2_pos = self._place_next_to(lvl1_pos)
        self.board[lvl2_pos[0]][lvl2_pos[1]] = 2
        lvl3_pos = self._place_next_to(lvl2_pos)
        self.board[lvl3_pos[0]][lvl3_pos[1]] = 3

        # Place the pawn randomly or next to the lvl1 tower
        if randint(0, 1) == 0:
            self.pawn_pos = self._get_random_different_than_tile(3)
        else:
            self.pawn_pos = self._place_next_to(lvl1_pos)
        self.nb_turns = 0

        return self.get_state()

    def get_state(self):
        state = []
        for i in range(5):
            for j in range(5):
                state.append(self.board[i][j])

        for i in range(5):
            for j in range(5):
                if (i, j) == self.pawn_pos:
                    state.append(1)
                else:
                    state.append(0)

        assert len(state) == num_states
        return state

    def get_game_score(self):
        return self.nb_turns - 2

    def step(self, action):
        # Play our player
        new_pos = get_new_pos_from_action(self.pawn_pos, action)

        current_height = self.board[self.pawn_pos[0]][self.pawn_pos[1]]
        self.pawn_pos = new_pos
        self.nb_turns += 1

        # Check if out of board
        if new_pos[0] < 0 or new_pos[0] > 4 or new_pos[1] < 0 or new_pos[1] > 4:
            # print(" /!\ Out of board at turn", self.nb_turns)
            self.nb_turns += 20
            return self.get_state(), -10, True

        # Check if moving too high
        new_height = self.board[new_pos[0]][new_pos[1]]
        if new_height > current_height + 1:
            # print(" !^! Moving too high at turn", self.nb_turns)
            self.nb_turns += 20
            return self.get_state(), -3, True

        # Check if reached lvl 3
        if new_height == 3:
            # print(" [+] Reached lvl 3 at turn", self.nb_turns)
            return self.get_state(), 10, True

        if self.nb_turns >= 10:
            print(" [!] Too many turns")
            return self.get_state(), -5, True

        return self.get_state(), new_height * 2, False

    def display(self):
        print("Turn", self.nb_turns)
        for i in range(5):
            for j in range(5):
                if (i, j) == self.pawn_pos:
                    print("X", end=" ")
                elif self.board[i][j] == 0:
                    print(".", end=" ")
                else:
                    print(self.board[i][j], end=" ")
            print()


if __name__ == "__main__":
    env = Env()
    env.reset()
    env.display()
    print(env.get_state())
