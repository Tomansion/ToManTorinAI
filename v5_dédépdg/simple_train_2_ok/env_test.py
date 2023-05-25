import numpy as np
from random import randint


class Env:
    def __init__(self):
        self.board_size = 5
        self.num_actions = 8
        self.num_states = (self.board_size * self.board_size) * 2 + self.num_actions

        self.move_count = [0] * self.num_actions
        self.nb_win = 0
        self.reset()

    def reset(self):
        self.pawn_position = [randint(0, 4), randint(0, 4)]
        self.goal_position = [randint(0, 4), randint(0, 4)]
        while self.goal_position == self.pawn_position:
            self.goal_position = [randint(0, 4), randint(0, 4)]

        self.nb_turn = 0

        return self._get_state()

    def step(self, action):
        row, col = self.pawn_position
        if action == 0:
            row -= 1
            col -= 1
        elif action == 1:
            row -= 1
        elif action == 2:
            row -= 1
            col += 1
        elif action == 3:
            col -= 1
        elif action == 4:
            col += 1
        elif action == 5:
            row += 1
            col -= 1
        elif action == 6:
            row += 1
        elif action == 7:
            row += 1
            col += 1

        self.pawn_position[0] = max(0, min(row, self.board_size - 1))
        self.pawn_position[1] = max(0, min(col, self.board_size - 1))

        self.move_count[action] += 1
        self.nb_turn += 1

        done = self.pawn_position == self.goal_position
        if done:
            reward = 1
            self.nb_win += 1
        else:
            reward = 0

        if self.nb_turn >= 10:
            done = True

        return self._get_state(), reward, done

    def _get_state(self):
        state = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.pawn_position == [i, j]:
                    state.append(1)
                else:
                    state.append(0)
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.goal_position == [i, j]:
                    state.append(1)
                else:
                    state.append(0)

        # Add if we can move in each direction
        for i in range(self.num_actions):
            if self.pawn_position[0] == 0 and i == 0:
                state.append(0)
            elif self.pawn_position[0] == self.board_size - 1 and i == 6:
                state.append(0)
            elif self.pawn_position[1] == 0 and i == 3:
                state.append(0)
            elif self.pawn_position[1] == self.board_size - 1 and i == 4:
                state.append(0)
            else:
                state.append(1)

        if len(state) != self.num_states:
            raise Exception("State size is not correct")
        return np.array(state)

    def display(self):
        print("Move count: {}".format(self.move_count))
        print("Number of wins: {}".format(self.nb_win))
        print("Board:")
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.pawn_position == [i, j]:
                    print("P", end=" ")
                elif self.goal_position == [i, j]:
                    print("G", end=" ")
                else:
                    print(".", end=" ")
            print("")
        print("")
