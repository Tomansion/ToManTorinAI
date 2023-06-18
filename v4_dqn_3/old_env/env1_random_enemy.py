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

# ======= DQN 3 ========
# Using only one model

# Env test, 10000 games, two random agents:
# nb win: 4645
# nb loose: 3497
# nb stuck self: 1325
# nb stuck other: 533

# After 1000 games of training:
# Average score over 3000 episodes: 0.8333333333333334
# nb win: 2500
# nb loose: 348
# nb stuck self: 79
# nb stuck other: 73

# After 5000 games of training:
# Average score over 3000 episodes: 0.9726666666666667
# nb win: 2918
# nb loose: 74
# nb stuck self: 3
# nb stuck other: 5

BOARD_SIZE = 4

NB_ACTIONS = 8 * 8  # 8 directions + 8 build

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

        # Stats
        self.nb_win = 0
        self.nb_loose = 0
        self.nb_stuck_self = 0
        self.nb_stuck_other = 0

        self.reset()

    def get_state_size(self):
        return NB_STATES

    def get_action_size(self):
        return NB_ACTIONS

    def reset(self):
        self.turn = 0

        # Place pawn
        self.pawn_pos = [randint(0, BOARD_SIZE - 1), randint(0, BOARD_SIZE - 1)]

        # Place enemy pawn
        self.enemy_pos = [randint(0, BOARD_SIZE - 1), randint(0, BOARD_SIZE - 1)]
        while self.enemy_pos == self.pawn_pos:
            self.enemy_pos = [randint(0, BOARD_SIZE - 1), randint(0, BOARD_SIZE - 1)]

        # Reset board
        self._reset_board()

        if randint(0, 1) == 0:
            self.move_enemy()

    def _reset_board(self):
        # Reset board
        self.board.fill(0)

        if not self.test:
            if randint(0, 1) == 0:
                return

            # Fill with random tiles
            nb_tiles = randint(0, BOARD_SIZE**3)
            for i in range(nb_tiles):
                pos = [randint(0, BOARD_SIZE - 1), randint(0, BOARD_SIZE - 1)]
                if self.board[pos[0]][pos[1]] <= 3:
                    self.board[pos[0]][pos[1]] += 1

            self.board[self.pawn_pos[0]][self.pawn_pos[1]] = randint(0, 2)
            self.board[self.enemy_pos[0]][self.enemy_pos[1]] = randint(0, 2)

    def move(self, action):
        self.turn += 1
        action = np.argmax(action)

        # DQN V3: 0 <= action <= 64 (8 * 8, move in 8 directions and build on 8 directions)
        move_action = action // 8
        build_action = action % 8

        # ====== Move ======
        new_pos = new_pos_from_action(self.pawn_pos, move_action)
        self.pawn_pos = new_pos

        # check goal
        new_level = self.board[self.pawn_pos[0]][self.pawn_pos[1]]
        if new_level == 3:
            self.nb_win += 1
            reward = 10
            return reward, True

        # ====== Build ======
        build_pos = new_pos_from_action(self.pawn_pos, build_action)
        self.board[build_pos[0]][build_pos[1]] += 1

        # Check stuck
        if self._is_stuck():
            self.nb_stuck_self += 1
            reward = -5
            return reward, True

        # ====== Enemy ======
        enemy_wins = self.move_enemy()
        if enemy_wins:
            self.nb_loose += 1
            reward = -10
            return reward, True

        # Check stuck
        if self._is_stuck():
            self.nb_stuck_other += 1
            reward = -5
            return reward, True

        return -1, False

    def move_enemy(self):
        def get_enemy_random_move():
            actions = np.arange(8)
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
            actions = np.arange(8)
            np.random.shuffle(actions)

            for action in actions:
                new_build_pos = new_pos_from_action(self.enemy_pos, action)
                if (
                    not is_outside(BOARD_SIZE, new_build_pos)
                    and new_build_pos != self.pawn_pos
                    and self.board[new_build_pos[0]][new_build_pos[1]] != 4
                ):
                    return new_build_pos
            return None

        # Move
        new_pos = get_enemy_random_move()
        if new_pos is None:
            # No move possible
            return

        self.enemy_pos = new_pos
        enemy_level = self.board[self.enemy_pos[0]][self.enemy_pos[1]]
        if enemy_level == 3:
            return True

        # Build
        build_pos = get_enemy_random_build()

        if build_pos is None:
            # No build possible
            self.render()
            raise Exception("No build possible for enemy")

        self.board[build_pos[0]][build_pos[1]] += 1

    def get_state(self, print_state=False):
        # Display a the board with the pawn in the middle
        state = [0] * NB_STATES
        state_id = 0

        # Add a layer for the board, 1 if in the board, 0 if outside
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

        # Add a layer for the enemy, 1 if enemy, 0 if not
        for j in range(-(FLASHLIGHT_SIZE - 1), FLASHLIGHT_SIZE):
            for i in range(-(FLASHLIGHT_SIZE - 1), FLASHLIGHT_SIZE):
                pos_rel = [self.pawn_pos[0] + i, self.pawn_pos[1] + j]
                if pos_rel == self.enemy_pos:
                    state[state_id] = 1
                state_id += 1

        # Add the player level to the state
        state[state_id] = self.board[self.pawn_pos[0]][self.pawn_pos[1]]
        # Add the enemy level to the state
        state[state_id + 1] = self.board[self.enemy_pos[0]][self.enemy_pos[1]]

        # Tell what actions are possible
        possible_actions = self._get_available_actions()

        if print_state:
            self.display_state(state)
            print("possible_actions:", possible_actions)

        return state, possible_actions

    def _is_stuck(self):
        for i in range(8):
            new_pos = new_pos_from_action(self.pawn_pos, i)
            if (
                is_tile_accessible(BOARD_SIZE, self.board, self.pawn_pos, new_pos)
                and new_pos != self.enemy_pos
            ):
                return False
        return True

    def _get_available_actions(self):
        possible_actions = [0] * NB_ACTIONS
        ## Move
        for i in range(8):
            new_pos = new_pos_from_action(self.pawn_pos, i)
            if (
                is_tile_accessible(BOARD_SIZE, self.board, self.pawn_pos, new_pos)
                and new_pos != self.enemy_pos
            ):
                # We can move at this position
                ## Build
                for j in range(8):
                    new_build_pos = new_pos_from_action(new_pos, j)
                    if (
                        not is_outside(BOARD_SIZE, new_build_pos)
                        and new_build_pos != self.enemy_pos
                    ):
                        # We can't build if the tile is at level 4 or if it's the enemy
                        if self.board[new_build_pos[0]][new_build_pos[1]] != 4:
                            possible_actions[8 * i + j] = 1

        return possible_actions

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
        print()

    def stats(self):
        print("nb win:", self.nb_win)
        print("nb loose:", self.nb_loose)
        print("nb stuck self:", self.nb_stuck_self)
        print("nb stuck other:", self.nb_stuck_other)


if __name__ == "__main__":
    from time import sleep

    user_input = False  # True if the user plays, False if random plays
    nb_games_todo = 10000
    nb_games = 0

    def get_random_action(possible_actions):
        # sleep(0.4)
        possible_action_numbers = []
        for i in range(len(possible_actions)):
            if possible_actions[i] == 1:
                possible_action_numbers.append(i)

        action = np.random.choice(possible_action_numbers)

        return action

    def get_user_action_from_state(type, state):
        action = int(input(type + " Action: "))
        return action

    # env = Env(test=False)
    env = Env(test=True)
    # Play the game with user input
    print("============")
    while True:
        state, possible_actions = env.get_state(print_state=False)
        # for i in range(NB_ACTIONS):
        #     if possible_moves[i]:
        #         mode_action_nb = i // 8
        #         build_action_nb = i % 8
        #         print("Move:", mode_action_nb, "Build:", build_action_nb)
        # print()
        # env.render()

        # actions = get_random_action_from_state(state)
        if user_input:
            move_action_nb = get_user_action_from_state("Move", state)
            build_action_nb = get_user_action_from_state("Build", state)

            # Build the 8*8 action vector
            actions = [0] * NB_ACTIONS
            actions[move_action_nb * 8 + build_action_nb] = 1
        else:
            action = get_random_action(possible_actions)
            actions = [0] * NB_ACTIONS
            actions[action] = 1

        reward, done = env.move(actions)
        print("reward:", reward)
        print("done:", done)

        if done:
            # env.render()
            env.stats()
            env.reset()
            nb_games += 1

            if nb_games == nb_games_todo:
                break
            # break
