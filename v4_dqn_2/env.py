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


# (prev) Env5: Enemy pawn (ง •̀_•́)ง
# - Feed corrected action to the model
# - Stop feeding possible actions to the model

# 1000 training: Average score over 3000 episodes: 1.304
# nb total score: 3912
# nb win: 2
# nb loose: 843
# nb out: 967
# nb long: 77
# nb nb_high: 736
# nb enemy: 138
# nb build out: 167
# nb build possible: 66
# nb build enemy: 4

# - -1 reward for each move

# Average score over 3000 episodes: 1.988
# nb total score: 5964
# nb win: 19
# nb loose: 893
# nb out: 1333
# nb long: 145
# nb nb_high: 508
# nb enemy: 12
# nb build out: 45
# nb build possible: 44
# nb build enemy: 1

# - Increase penalty for losing and -1 reward for each build
# Average score over 3000 episodes: 1.5316666666666667
# nb total score: 4595
# nb win: 13
# nb loose: 571
# nb out: 1291
# nb long: 55
# nb nb_high: 861
# nb enemy: 28
# nb build out: 54
# nb build possible: 121
# nb build enemy: 7

# - With board size 5 and flashlight size 4
# Average score over 3000 episodes: 0.7673333333333333
# nb total score: 2302
# nb win: 0
# nb loose: 395
# nb out: 2303
# nb long: 16
# nb nb_high: 268
# nb enemy: 12
# nb build out: 4
# nb build possible: 1
# nb build enemy: 1

# 5.6
# - Increase/Decrease reward for number of available moves and build
# - Decrease and game over if no available moves
# - Back to 4 size board
# Average score over 3000 episodes: 1.1476666666666666
# nb total score: 3443
# nb win: 0
# nb loose: 2072
# nb long: 262
# nb stuck: 20
# nb stuck build: 0
# nb out: 141
# nb nb_high: 278
# nb enemy: 47
# nb build out: 115
# nb build possible: 50
# nb build enemy: 15

# 5.7
# - Removed the reward change on available move
# 50000 train episodes
#   Best model:
# nb total score: 17938
# nb win: 1341
# nb loose: 438
# nb long: 1
# nb stuck: 52
# nb out: 971
# nb nb_high: 173
# nb enemy: 12
# nb build out: 9
# nb build possible: 3
# nb build enemy: 0
#   Last model :( :
# nb total score: 564
# nb win: 0
# nb loose: 972
# nb long: 6
# nb stuck: 25
# nb out: 1968
# nb nb_high: 23
# nb enemy: 4
# nb build out: 2
# nb build possible: 0
# nb build enemy: 0



# - Train with obvious moves


BOARD_SIZE = 4

NB_ACTIONS = 8

FLASHLIGHT_SIZE = BOARD_SIZE - 1
FLASHLIGHT_BORDER_SIZE = FLASHLIGHT_SIZE * 2 - 1
FLASHLIGHT_AREA = FLASHLIGHT_BORDER_SIZE**2

NB_STATES = FLASHLIGHT_AREA * 5  # Tiles
NB_STATES += FLASHLIGHT_AREA  # Enemy pawn position
NB_STATES += 1  # Player level
NB_STATES += 1  # Enemy level

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
        self.nb_enemy = 0
        self.nb_stuck = 0

        # Build
        self.nb_build_out = 0
        self.nb_build_possible = 0
        self.nb_build_enemy = 0

        # Enemy
        self.enemy_win = 0

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
        # Place enemy pawn
        self.enemy_pos = [randint(0, BOARD_SIZE - 1), randint(0, BOARD_SIZE - 1)]
        while self.enemy_pos == self.pawn_pos:
            self.enemy_pos = [randint(0, BOARD_SIZE - 1), randint(0, BOARD_SIZE - 1)]

        if not self.test:
            self.board[self.pawn_pos[0]][self.pawn_pos[1]] = randint(0, 2)
            self.board[self.enemy_pos[0]][self.enemy_pos[1]] = randint(0, 2)

        if randint(0, 1) == 0:
            self.move_enemy()

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
        reward = -1
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
        if not is_tile_accessible(BOARD_SIZE, self.board, self.pawn_pos, new_pos):
            game_over = True
            reward = -5
            self.nb_high += 1
            return reward, game_over

        # Check if tile is not occupied
        if new_pos == self.enemy_pos:
            game_over = True
            reward = -5
            self.nb_enemy += 1
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

        # Get nb possible moves
        available_moves = self._get_available_moves()
        nb_available_moves = sum(available_moves)

        if nb_available_moves == 0:
            game_over = True
            reward = -10
            self.nb_stuck += 1

        self.turn += 1

        return reward, game_over

    def build(self, action):
        reward = -1
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

        # Check if tile is not occupied
        if build_pos == self.enemy_pos:
            game_over = True
            reward = -5
            self.nb_build_enemy += 1
            return reward, game_over

        # Build
        self.board[build_pos[0]][build_pos[1]] += 1

        return reward, game_over

    def move_enemy(self):
        def get_enemy_random_move():
            actions = np.arange(NB_ACTIONS)
            np.random.shuffle(actions)

            for action in actions:
                new_pos = new_pos_from_action(self.enemy_pos, action)
                if (
                    is_tile_accessible(BOARD_SIZE, self.board, self.enemy_pos, new_pos)
                    and new_pos != self.pawn_pos
                ):
                    return new_pos
            return None

        def get_enemy_random_build():
            actions = np.arange(NB_ACTIONS)
            np.random.shuffle(actions)

            for action in actions:
                new_pos = new_pos_from_action(self.enemy_pos, action)
                if (
                    not is_outside(BOARD_SIZE, new_pos)
                    and new_pos != self.pawn_pos
                    and self.board[new_pos[0]][new_pos[1]] != 4
                ):
                    return new_pos
            return None

        # Move
        new_pos = get_enemy_random_move()
        if new_pos is None:
            # No move possible
            return

        self.enemy_pos = new_pos
        enemy_level = self.board[self.enemy_pos[0]][self.enemy_pos[1]]
        if enemy_level == 3:
            self.score -= 1
            self.total_score -= 1
            self.enemy_win += 1
            return True

        # Build
        build_pos = get_enemy_random_build()
        if build_pos is None:
            # No build possible
            return

        self.board[build_pos[0]][build_pos[1]] += 1

    def _get_board_state(self):
        # Display a the board with the pawn in the middle
        state = [0] * NB_STATES
        state_id = 0
        for tower_lever in range(5):
            for j in range(-(FLASHLIGHT_SIZE - 1), FLASHLIGHT_SIZE):
                for i in range(-(FLASHLIGHT_SIZE - 1), FLASHLIGHT_SIZE):
                    pos_rel = [self.pawn_pos[0] + i, self.pawn_pos[1] + j]
                    # print(i, j, pos_rel, state_id)

                    if is_outside(BOARD_SIZE, pos_rel):
                        state[state_id] = 0
                    else:
                        board_value = self.board[pos_rel[0]][pos_rel[1]]
                        if board_value == tower_lever:
                            state[state_id] = 1

                    state_id += 1

        # Add a layer for the enemy
        for j in range(-(FLASHLIGHT_SIZE - 1), FLASHLIGHT_SIZE):
            for i in range(-(FLASHLIGHT_SIZE - 1), FLASHLIGHT_SIZE):
                pos_rel = [self.pawn_pos[0] + i, self.pawn_pos[1] + j]
                if pos_rel == self.enemy_pos:
                    state[state_id] = 1
                state_id += 1

        # Add the player level to the state
        state[state_id] = self.board[self.pawn_pos[0]][self.pawn_pos[1]]
        state[state_id + 1] = self.board[self.enemy_pos[0]][self.enemy_pos[1]]

        return state

    def _get_available_moves(self):
        possible_actions = [0] * NB_ACTIONS
        for i in range(NB_ACTIONS):
            new_pos = new_pos_from_action(self.pawn_pos, i)
            if (
                is_tile_accessible(BOARD_SIZE, self.board, self.pawn_pos, new_pos)
                and new_pos != self.enemy_pos
            ):
                possible_actions[i] = 1
        return possible_actions

    def _get_available_build(self):
        possible_actions = [0] * NB_ACTIONS
        for i in range(NB_ACTIONS):
            new_pos = new_pos_from_action(self.pawn_pos, i)
            if not is_outside(BOARD_SIZE, new_pos) and new_pos != self.enemy_pos:
                # We can't build if the tile is at level 4 or if it's the enemy
                if self.board[new_pos[0]][new_pos[1]] != 4:
                    possible_actions[i] = 1

        return possible_actions

    def get_move_state(self, print_state=False):
        state = self._get_board_state()

        # Add if actions are possible
        possible_actions = self._get_available_moves()

        if print_state:
            self.display_state(state)

        return state, possible_actions

    def get_build_state(self, print_state=False):
        state = self._get_board_state()

        # Add if actions are possible
        possible_actions = self._get_available_build()

        if print_state:
            self.display_state(state)

        return state, possible_actions

    def render(self):
        print()
        for j in range(BOARD_SIZE - 1, -1, -1):
            for i in range(BOARD_SIZE):
                if self.pawn_pos == [i, j]:
                    print("👑", end="")
                elif self.enemy_pos == [i, j]:
                    print("👿", end="")
                else:
                    level = self.board[i][j]
                    if level == 0:
                        print("⬜", end="")
                    elif level == 1:
                        print("1️⃣", end=" ")
                    elif level == 2:
                        print("2️⃣", end=" ")
                    elif level == 3:
                        print("3️⃣", end=" ")
                    elif level == 4:
                        print("🔵", end="")
            print()
        print("Player level:", self.board[self.pawn_pos[0]][self.pawn_pos[1]])

    def display_state(self, state):
        for tower_lever in range(5):
            for j in range(FLASHLIGHT_BORDER_SIZE - 1, -1, -1):
                for i in range(FLASHLIGHT_BORDER_SIZE):
                    element = state[
                        j * FLASHLIGHT_BORDER_SIZE + i + tower_lever * FLASHLIGHT_AREA
                    ]
                    print(
                        element if element != -1 else "_",
                        end=" ",
                    )
                print()
            print()

        print()
        print("Ennemy level:", state[-(NB_ACTIONS + 1)])
        print("Player level:", state[-(NB_ACTIONS + 2)])
        # for i in range(NB_ACTIONS):
        #     if state[-NB_ACTIONS + i] == 1:
        #         print(" - Possible action:", i)
        print()

    def stats(self):
        print("nb total score:", self.total_score)
        print("nb win:", self.nb_win)
        print("nb loose:", self.enemy_win)
        print("nb long:", self.nb_long)
        print("nb stuck:", self.nb_stuck)
        print("nb out:", self.nb_out)
        print("nb nb_high:", self.nb_high)
        print("nb enemy:", self.nb_enemy)
        print("nb build out:", self.nb_build_out)
        print("nb build possible:", self.nb_build_possible)
        print("nb build enemy:", self.nb_build_enemy)


if __name__ == "__main__":
    from time import sleep

    def get_random_action_from_state(state):
        sleep(0.4)
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
        current_level = state[-2]
        print("current_level:", current_level)
        action = int(input(type + " Action: "))
        print("action:", action)
        actions = np.zeros(NB_ACTIONS)
        actions[action] = 1
        return actions

    # env = Env(test=False)
    env = Env(test=True)
    # Play the game with user input
    print("============")
    while True:
        # Move
        env.get_move_state(print_state=True)
        env.render()

        # Get action from the last 8 state values (the possible actions)
        state, _ = env.get_move_state()
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

        # Get action from the last 8 state values (the possible actions)
        state, _ = env.get_build_state()
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

        # Move enemy
        enemy_has_won = env.move_enemy()

        if enemy_has_won:
            env.render()
            env.stats()
            break
