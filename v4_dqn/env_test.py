from random import choice, randint
from utils import get_new_pos_from_action

# Test env

# 5x5 board
# goal: move across the board without going out
#       Reach the center of the board

num_states = 25
num_actions = 8


class Env:
    def __init__(self):
        self.reset()
        self.reward_goal = 30

    def reset(self):
        # Place the pawn randomly
        while True:
            self.pawn_pos = (randint(0, 4), randint(0, 4))
            if self.pawn_pos != (2, 2):
                break
        self.nb_turns = 0
        return self.get_state()

    def get_state(self):
        return self.get_state_from_pos(self.pawn_pos)

    def get_state_from_pos(self, pos):
        state = []
        for i in range(5):
            for j in range(5):
                if (i, j) == pos:
                    state.append(1)
                else:
                    state.append(0)
        return state

    def get_next_states(self):
        next_states = {}
        for action in range(num_actions):
            pos = get_new_pos_from_action(self.pawn_pos, action)
            next_states[action] = self.get_state_from_pos(pos)
        return next_states

    def get_game_score(self):
        return self.nb_turns - 2

    def step(self, action):
        # Play our player
        new_pos = get_new_pos_from_action(self.pawn_pos, action)

        self.pawn_pos = new_pos
        self.nb_turns += 1

        # Check if out of board
        if new_pos[0] < 0 or new_pos[0] > 4 or new_pos[1] < 0 or new_pos[1] > 4:
            print(" /!\ Out of board at turn", self.nb_turns)
            self.nb_turns += 20
            return self.get_state(), -10, True

        if new_pos == (2, 2):
            print(" <3 Goal reached at turn", self.nb_turns)
            return self.get_state(), 10, True

        if self.nb_turns >= 20:
            print(" [!] Too many turns")
            return self.get_state(), -10, True

        return self.get_state(), 0, False

    def display(self):
        print("Turn", self.nb_turns)
        for i in range(5):
            for j in range(5):
                if (i, j) == self.pawn_pos:
                    print("x", end="")
                elif (i, j) == (2, 2):
                    print("o", end="")
                else:
                    print(".", end="")
            print()
        print()