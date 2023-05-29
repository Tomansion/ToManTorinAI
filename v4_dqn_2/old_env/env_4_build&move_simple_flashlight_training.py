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

# Env4: Move and build solo, flashlight

# After 12000 games:
# Starting from empty board:
# Average score over 10000 episodes: 5.104
# nb total score: 51040
# nb win: 5104
# nb out: 0
# nb long: 0
# nb nb_high: 0
# nb build out: 4896
# nb build possible: 0

# Starting from train board:
# Average score over 10000 episodes: 0.3923
# nb total score: 3923
# nb win: 0
# nb out: 0
# nb long: 0
# nb nb_high: 1340
# nb build out: 8348
# nb build possible: 312

# After 12000 games:
# Starting from empty board:
# Average score over 10000 episodes: 2.444
# nb total score: 24440
# nb win: 2444
# nb out: 0
# nb long: 0
# nb nb_high: 0
# nb build out: 7556
# nb build possible: 0

# Starting from train board:
# Average score over 10000 episodes: 0.5402
# nb total score: 5402
# nb win: 1
# nb out: 0
# nb long: 0
# nb nb_high: 306
# nb build out: 9490
# nb build possible: 203

# Full random test:
# Test env:
# Average score over 10000 episodes: 0.0003
# nb total score: 3
# nb win: 0
# nb out: 5392
# nb long: 0
# nb nb_high: 320
# nb build out: 4288
# nb build possible: 0
# Training env:
# Testing random...
# Average score over 10000 episodes: 0.039
# nb total score: 390
# nb win: 0
# nb out: 4193
# nb long: 0
# nb nb_high: 2719
# nb build out: 2709
# nb build possible: 379

# Full no mistake test:
# Test env:
# Average score over 10000 episodes: 1.4671
# nb total score: 14671
# nb win: 0
# nb out: 2808
# nb long: 5719
# nb nb_high: 1473
# nb build out: 0
# nb build possible: 0
# Training env:
# Average score over 10000 episodes: 2.7068
# nb total score: 27068
# nb win: 27
# nb out: 3690
# nb long: 3631
# nb nb_high: 2630
# nb build out: 17
# nb build possible: 7

# Second training got worse ¯\_(ツ)_/¯ 
# Test env:
# Average score over 10000 episodes: 0.5171
# nb total score: 5171
# nb win: 0
# nb out: 0
# nb long: 0
# nb nb_high: 4828
# nb build out: 0
# nb build possible: 5172
# Train env:
# Average score over 10000 episodes: 0.8461
# nb total score: 8461
# nb win: 4
# nb out: 5
# nb long: 0
# nb nb_high: 8019
# nb build out: 27
# nb build possible: 1945




NB_ACTIONS = 8
NB_STATES = (BOARD_SIZE * 2 - 1) ** 2 + NB_ACTIONS


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
        if is_outside(self.board, new_pos):
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
        if is_outside(self.board, build_pos):
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

    def get_move_state(self, print_state=False):
        # Display a the board with the pawn in the middle
        state = []
        for j in range(-4, 5):
            for i in range(-4, 5):
                pos_rel = [self.pawn_pos[0] + i, self.pawn_pos[1] + j]

                if is_outside(self.board, pos_rel):
                    state.append(-1)
                else:
                    board_value = self.board[pos_rel[0]][pos_rel[1]]
                    state.append(board_value)

        if print_state:
            for j in range(8 - 1, -1, -1):
                for i in range(9):
                    print(state[j * 9 + i], end=" ")
                print()
            print()

        # Add if actions are possible
        for i in range(NB_ACTIONS):
            new_pos = new_pos_from_action(self.pawn_pos, i)
            if is_outside(self.board, new_pos):
                state.append(0)
            elif not is_tile_accessible(self.board, self.pawn_pos, new_pos):
                state.append(0)
            else:
                state.append(1)

            if print_state:
                print(str(i) + ":", state[-1])

        if len(state) != NB_STATES:
            raise Exception("state size is not correct")
        return state

    def get_build_state(self, print_state=False):
        # Display a the board with the pawn in the middle
        state = []
        for j in range(-4, 5):
            for i in range(-4, 5):
                pos_rel = [self.pawn_pos[0] + i, self.pawn_pos[1] + j]

                if is_outside(self.board, pos_rel):
                    state.append(-1)
                else:
                    board_value = self.board[pos_rel[0]][pos_rel[1]]
                    state.append(board_value)

        if print_state:
            for j in range(8 - 1, -1, -1):
                for i in range(9):
                    print(state[j * 9 + i], end=" ")
                print()
            print()

        # Add if actions are possible
        for i in range(NB_ACTIONS):
            new_pos = new_pos_from_action(self.pawn_pos, i)
            if is_outside(self.board, new_pos):
                state.append(0)
            else:
                pos_level = self.board[new_pos[0]][new_pos[1]]
                if pos_level == 4:
                    state.append(0)
                else:
                    state.append(1)

            if print_state:
                print(str(i) + ":", state[-1])

        if len(state) != NB_STATES:
            raise Exception("state size is not correct")
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

    env = Env(test=False)
    # Play the game with user input
    print("============")
    while True:
        # Move
        env.get_move_state(print_state=True)
        env.render()
        sleep(0.4)
        # action = int(input("Move action: "))

        # Get action from the last 8 state values (the possible actions)
        state = env.get_move_state()
        actions = get_random_action_from_state(state)
        reward, done = env.move(actions)
        env.get_move_state(print_state=True)
        print("reward:", reward)
        print("done:", done)
        print("score:", env.score)

        if done:
            env.render()
            env.stats()
            break

        # Build
        env.get_build_state(print_state=True)
        env.render()
        sleep(0.4)

        # action = int(input("Build action: "))
        # Get action from the last 8 state values (the possible actions)
        state = env.get_build_state()
        actions = get_random_action_from_state(state)
        reward, done = env.build(actions)
        env.get_build_state(print_state=True)

        print("reward:", reward)
        print("done:", done)
        print("score:", env.score)
        if done:
            env.render()
            env.stats()
            break
