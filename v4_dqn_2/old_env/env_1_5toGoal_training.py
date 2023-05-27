from random import randint
import numpy as np

BOARD_SIZE = 5

# Env1 : Simple goal chase

# NB_STATES = (BOARD_SIZE * 2 - 1) ** 2
NB_STATES = ((BOARD_SIZE) ** 2) * 2
NB_ACTIONS = 4


class Env:
    def __init__(self):
        self.reset()

    def get_state_size(self):
        return NB_STATES

    def get_action_size(self):
        return NB_ACTIONS

    def reset(self):
        self.pawn_pos = [randint(0, BOARD_SIZE - 1), randint(0, BOARD_SIZE - 1)]
        self._place_goal()

        self.score = 0

    def _place_goal(self):
        self.goal_pos = [randint(0, BOARD_SIZE - 1), randint(0, BOARD_SIZE - 1)]
        while self.goal_pos == self.pawn_pos:
            self.goal_pos = [randint(0, BOARD_SIZE - 1), randint(0, BOARD_SIZE - 1)]

    def step(self, action):
        action = np.argmax(action)
        # 2. move
        if action == 0:  # right
            self.pawn_pos[0] += 1
        elif action == 1:  # left
            self.pawn_pos[0] -= 1
        elif action == 2:  # up
            self.pawn_pos[1] -= 1
        elif action == 3:  # down
            self.pawn_pos[1] += 1

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_outside(self.pawn_pos):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new goal
        if self.pawn_pos == self.goal_pos:
            self.score += 1
            reward = 10
            self._place_goal()

        if self.score == 10:
            game_over = True

        return reward, game_over, self.score

    def is_outside(self, coord):
        return (
            coord[0] > BOARD_SIZE - 1
            or coord[0] < 0
            or coord[1] > BOARD_SIZE - 1
            or coord[1] < 0
        )

    def get_state(self):
        # Display a the board with the pawn in the middle
        state_pawn = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        if not self.is_outside(self.pawn_pos):
            state_pawn[self.pawn_pos[0], self.pawn_pos[1]] = 1
        state_pawn = state_pawn.flatten()
        state_goal = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        state_goal[self.goal_pos[0], self.goal_pos[1]] = 1
        state_goal = state_goal.flatten()
        state = np.concatenate((state_pawn, state_goal))
        return state

    def render(self):
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.pawn_pos == [i, j]:
                    print("P", end=" ")
                elif self.goal_pos == [i, j]:
                    print("G", end=" ")
                else:
                    print("_", end=" ")
            print()
        print()
