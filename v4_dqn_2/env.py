from random import randint
import numpy as np
from helper import (
    new_pos_from_action,
    get_random_empty_tile,
    get_random_different_than_tile,
    place_next_to,
    is_tile_accessible,
    create_empty_board,
    is_outside,
)

BOARD_SIZE = 5

# Env1 : reach T2, flashlight

# Stage : centered on the pawn

# Average score over 1000 episodes: 1.223
# nb win: 1
# nb out: 0
# nb long: 112
# nb nb_high: 887
# Training slowly, sees to be confusing
# the two towers
# After training 8000 episodes, sudden improvement
# After training 10000 episodes : 6 win in average
# Average score over 10000 episodes: 4.9221
# nb win: 2033
# nb out: 0
# nb long: 2894
# nb nb_high: 5073

NB_STATES = (BOARD_SIZE * 2 - 1) ** 2
NB_ACTIONS = 8


class Env:
    def __init__(self):
        self.reset()
        self.nb_win = 0
        self.nb_out = 0
        self.nb_long = 0
        self.nb_high = 0

    def get_state_size(self):
        return NB_STATES

    def get_action_size(self):
        return NB_ACTIONS

    def reset(self):
        self.pawn_pos = None
        self._place_goal()

        r = randint(0, 2)
        if r == 0:
            self.pawn_pos = self.t1_pos
        elif r == 1:
            self.pawn_pos = place_next_to(self.board, self.t1_pos)
        else:
            self.pawn_pos = get_random_empty_tile(self.board)

        self.score = 0
        self.turn = 0

    def _place_goal(self):
        # Reset board
        self.board = create_empty_board(BOARD_SIZE)

        self.t2_pos = get_random_empty_tile(self.board)
        if self.pawn_pos is not None:
            while self.t2_pos == self.pawn_pos:
                self.t2_pos = get_random_empty_tile(self.board)

        self.board[self.t2_pos[0]][self.t2_pos[1]] = 2
        self.t1_pos = place_next_to(self.board, self.t2_pos)
        self.board[self.t1_pos[0]][self.t1_pos[1]] = 1

    def step(self, action):
        action = np.argmax(action)
        new_pos = new_pos_from_action(self.pawn_pos, action)

        # check if out of bounds
        reward = 0
        game_over = False
        if is_outside(self.board, new_pos):
            game_over = True
            reward = -10
            self.nb_out += 1
            return reward, game_over, self.score

        # Check if tile is accessible
        if not is_tile_accessible(self.board, self.pawn_pos, new_pos):
            game_over = True
            reward = -5
            self.nb_high += 1
            return reward, game_over, self.score

        self.pawn_pos = new_pos

        # check goal
        if self.pawn_pos == self.t2_pos:
            self.score += 1
            reward = 10
            self._place_goal()

        if self.score == 10:
            game_over = True
            self.nb_win += 1

        if self.turn == 100:
            game_over = True
            self.nb_long += 1

        self.turn += 1

        return reward, game_over, self.score

    def get_state(self, print_state=False):
        # Display a the board with the pawn in the middle
        state = []
        for j in range(-4, 5):
            for i in range(-4, 5):
                pos_rel = [self.pawn_pos[0] + i, self.pawn_pos[1] + j]
                if is_outside(self.board, pos_rel):
                    state.append(-1)
                elif pos_rel == self.t2_pos:
                    state.append(2)
                elif pos_rel == self.t1_pos:
                    state.append(1)
                else:
                    state.append(0)

        if print_state:
            for j in range(8 - 1, -1, -1):
                for i in range(9):
                    print(state[j * 9 + i], end=" ")
                print()
            print()

        if len(state) != NB_STATES:
            raise Exception("state size is not correct")
        return state

    def render(self):
        for j in range(BOARD_SIZE - 1, -1, -1):
            for i in range(BOARD_SIZE):
                if self.pawn_pos == [i, j]:
                    print("P", end=" ")
                elif self.t2_pos == [i, j]:
                    print("2", end=" ")
                elif self.t1_pos == [i, j]:
                    print("1", end=" ")
                else:
                    print("_", end=" ")
            print()
        print()

    def stats(self):
        print("nb win:", self.nb_win)
        print("nb out:", self.nb_out)
        print("nb long:", self.nb_long)
        print("nb nb_high:", self.nb_high)


if __name__ == "__main__":
    env = Env()
    env.get_state(print_state=True)
    env.render()

    # Play the game with user input
    print("============")
    while True:
        action = int(input("action: "))
        reward, done, score = env.step(action)
        env.get_state(print_state=True)
        env.render()
        print("reward:", reward)
        print("done:", done)
        print("score:", score)
        if done:
            break
