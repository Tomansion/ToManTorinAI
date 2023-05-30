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


# Env4:3 Multiple smaller flashlight with level to state,
# Env5: Enemy pawn (ง •̀_•́)ง
#    - Alwars start from empty board
#    - Negative reward on step

# After 40000 episodes:
#   Average score over 10000 episodes: 3.6537
#   nb total score: 36537
#   nb win: 1134
#   nb loose: 2185
#   nb out: 5
#   nb long: 316
#   nb nb_high: 1466
#   nb enemy: 494
#   nb build out: 52
#   nb build possible: 3856
#   nb build enemy: 515

# TODO: add height difference (-3 to 4) to state

BOARD_SIZE = 4

NB_ACTIONS = 8

FLASHLIGHT_SIZE = BOARD_SIZE - 1
FLASHLIGHT_BORDER_SIZE = FLASHLIGHT_SIZE * 2 - 1
FLASHLIGHT_AREA = FLASHLIGHT_BORDER_SIZE**2

NB_STATES = FLASHLIGHT_AREA * 5  # Tiles
NB_STATES += FLASHLIGHT_AREA  # Enemy pawn position
NB_STATES += 1  # Player level
NB_STATES += 1  # Enemy level
NB_STATES += NB_ACTIONS  # Possible actions

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

    def get_move_state(self, print_state=False):
        state = self._get_board_state()

        # Add if actions are possible
        for i in range(NB_ACTIONS):
            new_pos = new_pos_from_action(self.pawn_pos, i)
            if (
                is_tile_accessible(BOARD_SIZE, self.board, self.pawn_pos, new_pos)
                and new_pos != self.enemy_pos
            ):
                state[-NB_ACTIONS + i] = 1

        if print_state:
            self.display_state(state)

        return state

    def get_build_state(self, print_state=False):
        state = self._get_board_state()

        # Add if actions are possible
        for i in range(NB_ACTIONS):
            new_pos = new_pos_from_action(self.pawn_pos, i)
            if not is_outside(BOARD_SIZE, new_pos) and new_pos != self.enemy_pos:
                # We can't build if the tile is at level 4 or if it's the enemy
                if self.board[new_pos[0]][new_pos[1]] != 4:
                    state[-NB_ACTIONS + i] = 1

        if print_state:
            self.display_state(state)

        return state

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
        for i in range(NB_ACTIONS):
            if state[-NB_ACTIONS + i] == 1:
                print(" - Possible action:", i)
        print()

    def stats(self):
        print("nb total score:", self.total_score)
        print("nb win:", self.nb_win)
        print("nb loose:", self.enemy_win)
        print("nb out:", self.nb_out)
        print("nb long:", self.nb_long)
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
        possible_actions = state[-NB_ACTIONS:]
        current_level = state[-NB_ACTIONS - 1]
        print("current_level:", current_level)
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

    # env = Env(test=False)
    env = Env(test=True)
    # Play the game with user input
    print("============")
    while True:
        # Move
        env.get_move_state(print_state=True)
        env.render()

        # Get action from the last 8 state values (the possible actions)
        state = env.get_move_state()
        actions = get_random_action_from_state(state)
        # actions = get_user_action_from_state("Move", state)

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
        state = env.get_build_state()
        actions = get_random_action_from_state(state)
        # actions = get_user_action_from_state("Build", state)
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

