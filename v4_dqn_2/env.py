from random import randint
import numpy as np
from helper import (
    VEC_MAP,
    new_pos_from_action,
    get_random_empty_tile,
    get_random_different_than_tile,
    place_next_to,
    is_tile_accessible,
    create_empty_board,
    is_outside,
)

BOARD_SIZE = 5


# TODO: change the pos on fail
# TODO: decrease flashlight size
# TODO: Only have 0 and one in the flashlight

# Env4:2: Move and build solo, flashlight * 5 for each tower state
# Multiple flashlight : Not training
# Multiple flashlight with bigger model : Not training


NB_ACTIONS = 8

FLASHLIGHT_SIZE = BOARD_SIZE * 2 - 1
FLASHLIGHT_AREA = FLASHLIGHT_SIZE**2

NB_STATES = FLASHLIGHT_AREA * 5 + NB_ACTIONS
print("NB_STATES:", NB_STATES)


class Env:
    def __init__(self, test=False):
        self.test = test
        self.board = create_empty_board(BOARD_SIZE)
        self.nb_win = 0
        self.total_score = 0

        # Move
        self.nb_out = 0
        self.nb_long = 0
        self.nb_high = 0

        # Build
        self.nb_build_out = 0
        self.nb_build_possible = 0

        self.reset()

    def get_state_size(self):
        return NB_STATES

    def get_action_size(self):
        return NB_ACTIONS

    def reset(self):
        self.score = 0
        self.turn = 0
        # Reset board
        self._reset_board()

        # Place pawn
        self.pawn_pos = [randint(0, BOARD_SIZE - 1), randint(0, BOARD_SIZE - 1)]
        if not self.test:
            self.board[self.pawn_pos[0]][self.pawn_pos[1]] = randint(0, 2)

    def _reset_board(self):
        # Reset board
        self.board.fill(0)

        if not self.test:
            # Fill with random tiles
            nb_tiles = max(0, randint(0, BOARD_SIZE * 15) - 10)
            for i in range(nb_tiles):
                pos = [randint(0, BOARD_SIZE - 1), randint(0, BOARD_SIZE - 1)]
                if self.board[pos[0]][pos[1]] <= 3:
                    self.board[pos[0]][pos[1]] += 1

    def move(self, action):
        reward = 0
        game_over = False
        action = np.argmax(action)
        new_pos = new_pos_from_action(self.pawn_pos, action)

        # check if out of bounds
        if is_outside(BOARD_SIZE, new_pos):
            game_over = True
            reward = -10
            self.nb_out += 1
            return reward, game_over

        # Check if tile is accessible
        if not is_tile_accessible(self.board, self.pawn_pos, new_pos):
            game_over = True
            reward = -5
            self.nb_high += 1
            return reward, game_over

        # Move pawn
        self.pawn_pos = new_pos

        # check goal
        new_level = self.board[self.pawn_pos[0]][self.pawn_pos[1]]
        if new_level == 3:
            self.score += 1
            self.total_score += 1
            reward = 10
            self._reset_board()

        if self.score == 10:
            game_over = True
            self.nb_win += 1

        if self.turn == 100:
            game_over = True
            self.nb_long += 1

        self.turn += 1

        return reward, game_over

    def build(self, action):
        reward = 0
        game_over = False
        action = np.argmax(action)
        build_pos = new_pos_from_action(self.pawn_pos, action)

        # check if out of bounds
        if is_outside(BOARD_SIZE, build_pos):
            game_over = True
            reward = -10
            self.nb_build_out += 1
            return reward, game_over

        # Check if build is possible
        build_level = self.board[build_pos[0]][build_pos[1]]
        if build_level == 4:
            game_over = True
            reward = -5
            self.nb_build_possible += 1
            return reward, game_over

        # Build
        self.board[build_pos[0]][build_pos[1]] += 1

        return reward, game_over

    def _get_board_state(self, print_state=False):
        # Display a the board with the pawn in the middle
        state = [0] * NB_STATES
        state_id = 0
        for tower_lever in range(5):
            for j in range(-4, 5):
                for i in range(-4, 5):
                    pos_rel = [self.pawn_pos[0] + i, self.pawn_pos[1] + j]

                    if is_outside(BOARD_SIZE, pos_rel):
                        state[state_id] = -1
                    else:
                        board_value = self.board[pos_rel[0]][pos_rel[1]]
                        if board_value == tower_lever:
                            state[state_id] = 1

                    state_id += 1

        if print_state:
            for tower_lever in range(5):
                for j in range(8 - 1, -1, -1):
                    for i in range(9):
                        print(state[j * 9 + i + tower_lever * FLASHLIGHT_AREA], end=" ")
                    print()
                print()

        return state

    def get_move_state(self, print_state=False):
        state = self._get_board_state(print_state)

        # Add if actions are possible
        for i in range(NB_ACTIONS):
            new_pos = new_pos_from_action(self.pawn_pos, i)
            if is_tile_accessible(self.board, self.pawn_pos, new_pos):
                state[-NB_ACTIONS + i] = 1

            if print_state and state[-1] == 1:
                print(str(i), end=", ")

        return state

    def get_build_state(self, print_state=False):
        state = self._get_board_state(print_state)

        # Add if actions are possible
        for i in range(NB_ACTIONS):
            new_pos = new_pos_from_action(self.pawn_pos, i)
            if not is_outside(BOARD_SIZE, new_pos):
                # We can't build if the tile is at level 4
                if self.board[new_pos[0]][new_pos[1]] != 4:
                    state[-NB_ACTIONS + i] = 1

            if print_state and state[-1] == 1:
                print(str(i), end=", ")

        if print_state:
            print()

        return state

    def render(self):
        print()
        for j in range(BOARD_SIZE - 1, -1, -1):
            for i in range(BOARD_SIZE):
                if self.pawn_pos == [i, j]:
                    print("P", end=" ")
                else:
                    level = self.board[i][j]
                    if level == 0:
                        print("_", end=" ")
                    else:
                        print(level, end=" ")
            print()
        print("Player level:", self.board[self.pawn_pos[0]][self.pawn_pos[1]])

    def stats(self):
        print("nb total score:", self.total_score)
        print("nb win:", self.nb_win)
        print("nb out:", self.nb_out)
        print("nb long:", self.nb_long)
        print("nb nb_high:", self.nb_high)
        print("nb build out:", self.nb_build_out)
        print("nb build possible:", self.nb_build_possible)


if __name__ == "__main__":
    from time import sleep

    def get_random_action_from_state(state):
        possible_actions = state[-NB_ACTIONS:]
        possible_action_numbers = []
        for i in range(len(possible_actions)):
            if possible_actions[i] == 1:
                possible_action_numbers.append(i)
        if len(possible_action_numbers) == 0:
            action = 0
        else:
            action = np.random.choice(possible_action_numbers)
        print("action:", action)
        actions = np.zeros(NB_ACTIONS)
        actions[action] = 1
        return actions

    def get_user_action_from_state(type, state):
        possible_actions = state[-NB_ACTIONS:]
        print()
        for i in range(len(possible_actions)):
            if possible_actions[i] == 1:
                print(" - ", i, VEC_MAP[i])
        print()
        action = int(input(type + " Action: "))
        print("action:", action)
        actions = np.zeros(NB_ACTIONS)
        actions[action] = 1
        return actions

    env = Env(test=False)
    # Play the game with user input
    print("============")
    while True:
        # Move
        env.get_move_state(print_state=True)
        env.render()
        sleep(0.4)

        # Get action from the last 8 state values (the possible actions)
        state = env.get_move_state()
        # actions = get_random_action_from_state(state)
        actions = get_user_action_from_state("Move", state)

        reward, done = env.move(actions)
        env.get_move_state(print_state=True)
        print("reward:", reward)
        print("done:", done)
        print("score:", env.score)

        if done:
            env.render()
            env.stats()
            break

        if reward > 0:
            continue

        # Build
        env.get_build_state(print_state=True)
        env.render()
        sleep(0.4)

        # Get action from the last 8 state values (the possible actions)
        state = env.get_build_state()
        # actions = get_random_action_from_state(state)
        actions = get_user_action_from_state("Build", state)
        reward, done = env.build(actions)
        env.get_build_state(print_state=True)

        print("reward:", reward)
        print("done:", done)
        print("score:", env.score)
        if done:
            env.render()
            env.stats()
            break
