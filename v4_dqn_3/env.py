from random import randint, choice
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
    action_from_pos,
)

from santorinai.board import Board
from santorinai.player_examples import random_player, basic_player, first_choice_player


# ======= DQN 3 ========
# Env2, implementing santorinai lib
# Afte 12000 episodes: of training:
# = Against random player
# Average score over 3000 episodes: 0.8253333333333334
# nb win: 2476
# nb loose: 247
# nb stuck self: 131
# nb stuck other: 146
# = Against First choice player:
# Average score over 3000 episodes: 0.33366666666666667
# nb win: 1001
# nb loose: 1761
# nb stuck self: 165
# nb stuck other: 73
# = Against Basic player
# Average score over 3000 episodes: 0.15133333333333332
# nb win: 454
# nb loose: 2450
# nb stuck self: 27
# nb stuck other: 69

BOARD_SIZE = 5

NB_ACTIONS = 8 * 8  # 8 directions + 8 build

FLASHLIGHT_SIZE = BOARD_SIZE - 1
FLASHLIGHT_BORDER_SIZE = FLASHLIGHT_SIZE * 2 - 1
FLASHLIGHT_AREA = FLASHLIGHT_BORDER_SIZE**2

NB_STATES = FLASHLIGHT_AREA * 5  # Tiles
NB_STATES += FLASHLIGHT_AREA  # Enemy pawn position
NB_STATES += 4  # Pawns level

print("NB_STATES:", NB_STATES)

possible_enemy = [
    basic_player.BasicPlayer(),
    first_choice_player.FirstChoicePlayer(),
    random_player.RandomPlayer(),
]


class Env:
    def __init__(self, test=False):
        self.test = test
        self.playing_pawn = None

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
        # Init board
        self.board = Board(2)

        # Select random enemy
        # self.enemy = first_choice_player.FirstChoicePlayer()
        self.enemy = random_player.RandomPlayer()


        def random_pos():
            return (randint(0, BOARD_SIZE - 1), randint(0, BOARD_SIZE - 1))

        # Place pawns
        pawn_pos_1 = random_pos()
        pawn_pos_2 = random_pos()
        while pawn_pos_1 == pawn_pos_2:
            pawn_pos_2 = random_pos()

        self.board.pawns[0].pos = pawn_pos_1
        self.board.pawns[2].pos = pawn_pos_2

        # Place enemy pawns
        enemy_pos_1 = random_pos()
        while enemy_pos_1 == pawn_pos_1 or enemy_pos_1 == pawn_pos_2:
            enemy_pos_1 = random_pos()
        enemy_pos_2 = random_pos()
        while (
            enemy_pos_2 == pawn_pos_1
            or enemy_pos_2 == pawn_pos_2
            or enemy_pos_2 == enemy_pos_1
        ):
            enemy_pos_2 = random_pos()

        self.board.pawns[1].pos = enemy_pos_1
        self.board.pawns[3].pos = enemy_pos_2

        # Place random tiles
        if not self.test:
            if randint(0, 1) == 0:
                return

            # Fill with random tiles
            nb_tiles = randint(0, BOARD_SIZE**2 * 2)
            for i in range(nb_tiles):
                pos = [randint(0, BOARD_SIZE - 1), randint(0, BOARD_SIZE - 1)]
                if self.board.board[pos[0]][pos[1]] <= 3:
                    self.board.board[pos[0]][pos[1]] += 1

            self.board.board[pawn_pos_1[0]][pawn_pos_1[1]] = randint(0, 2)
            self.board.board[pawn_pos_2[0]][pawn_pos_2[1]] = randint(0, 2)
            self.board.board[enemy_pos_1[0]][enemy_pos_1[1]] = randint(0, 2)
            self.board.board[enemy_pos_2[0]][enemy_pos_2[1]] = randint(0, 2)

        # Don't always start first
        if randint(0, 1) == 0:
            enemy_won = self.move_enemy()
            if enemy_won:
                self.reset()

    def move(self, action):
        action = np.argmax(action)

        move_action = int(action // 8)
        build_action = int(action % 8)

        self.playing_pawn = self.board.get_playing_pawn()

        new_move_pos = new_pos_from_action(self.playing_pawn.pos, move_action)
        new_build_pos = new_pos_from_action(new_move_pos, build_action)

        # ====== Move ======
        self.board.play_move(new_move_pos, new_build_pos)

        # check goal
        if self.board.is_game_over():
            self.nb_win += 1
            reward = 10
            return reward, True

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
        if self.enemy is None:
            return False

        # Get action
        board_copy = self.board.copy()
        move_pos, build_pos = self.enemy.play_move(
            board_copy, board_copy.get_playing_pawn()
        )

        if move_pos is None or build_pos is None:
            # No move possible
            return

        # Apply move
        self.board.play_move(move_pos, build_pos)

        if self.board.is_game_over():
            return True

    def get_state(self, print_state=False):
        # Display a the board with the pawn in the middle
        state = [0] * NB_STATES
        state_id = 0

        playing_pawn = self.board.get_playing_pawn()
        playing_pawn_nb = playing_pawn.number

        center_pos = [playing_pawn.pos[0], playing_pawn.pos[1]]
        if center_pos[0] == None or center_pos[1] == None:
            raise Exception("Pawn position is None")

        # Add a layer for the board, 1 if in the board, 0 if outside
        for tower_lever in range(5):
            for j in range(-(FLASHLIGHT_SIZE - 1), FLASHLIGHT_SIZE):
                for i in range(-(FLASHLIGHT_SIZE - 1), FLASHLIGHT_SIZE):
                    pos_rel = [center_pos[0] + i, center_pos[1] + j]
                    # print(i, j, pos_rel, state_id)

                    if is_outside(BOARD_SIZE, pos_rel):
                        state[state_id] = 0
                    else:
                        board_value = self.board.board[pos_rel[0]][pos_rel[1]]
                        if board_value == tower_lever:
                            state[state_id] = 1

                    state_id += 1

        # Add a layer for the enemy, 1 if enemy 1, 2 if enemy 2, 0 if not
        ally_pawn_nb = (playing_pawn_nb + 2) % 4
        ally_pawn = self.board.pawns[ally_pawn_nb - 1]
        enemy_1_pawn_nb = (playing_pawn_nb + 1) % 4
        enemy_2_pawn_nb = (playing_pawn_nb + 3) % 4
        enemy_1_pawn = self.board.pawns[enemy_1_pawn_nb - 1]
        enemy_2_pawn = self.board.pawns[enemy_2_pawn_nb - 1]
        for j in range(-(FLASHLIGHT_SIZE - 1), FLASHLIGHT_SIZE):
            for i in range(-(FLASHLIGHT_SIZE - 1), FLASHLIGHT_SIZE):
                pos_rel = [center_pos[0] + i, center_pos[1] + j]
                if pos_rel == enemy_1_pawn.pos:
                    state[state_id] = 1
                if pos_rel == enemy_2_pawn.pos:
                    state[state_id] = 2
                if pos_rel == ally_pawn.pos:
                    state[state_id] = 3
                state_id += 1

        # Add the player and ally level to the state
        state[state_id] = self.board.board[center_pos[0]][center_pos[1]]
        state[state_id + 1] = self.board.board[ally_pawn.pos[0]][ally_pawn.pos[1]]
        # Add the enemy level to the state
        state[state_id + 2] = self.board.board[enemy_1_pawn.pos[0]][enemy_1_pawn.pos[1]]
        state[state_id + 3] = self.board.board[enemy_2_pawn.pos[0]][enemy_2_pawn.pos[1]]

        # Tell what actions are possible
        possible_actions = self._get_available_actions()

        if print_state:
            self.display_state(state)
            print("possible_actions:", possible_actions)

        return state, possible_actions

    def _is_stuck(self):
        playing_pawn = self.board.get_playing_pawn()
        possible_actions = self.board.get_possible_movement_and_building_positions(
            playing_pawn
        )
        return len(possible_actions) == 0

    def _get_available_actions(self):
        playing_pawn = self.board.get_playing_pawn()
        possible_moves = self.board.get_possible_movement_and_building_positions(
            playing_pawn
        )

        # Possible actions: [(move_pos, build_pos), ...]
        # Convert to a list of actions

        possible_actions = [0] * NB_ACTIONS

        for move_pos, build_pos in possible_moves:
            if move_pos is None or build_pos is None:
                continue

            move_action = action_from_pos(playing_pawn.pos, move_pos)
            build_action = action_from_pos(move_pos, build_pos)

            if move_action is None or build_action is None:
                continue

            action = move_action * 8 + build_action
            possible_actions[action] = 1

        return possible_actions

    def render(self):
        print()

        playing_pawn = self.board.get_playing_pawn()
        playing_pawn_nb = playing_pawn.number
        ally_pawn_nb = (playing_pawn_nb + 2) % 4
        ally_pawn = self.board.pawns[ally_pawn_nb - 1]
        enemy_1_pawn_nb = (playing_pawn_nb + 1) % 4
        enemy_2_pawn_nb = (playing_pawn_nb + 3) % 4
        enemy_1_pawn = self.board.pawns[enemy_1_pawn_nb - 1]
        enemy_2_pawn = self.board.pawns[enemy_2_pawn_nb - 1]

        for j in range(BOARD_SIZE - 1, -1, -1):
            for i in range(BOARD_SIZE):
                if playing_pawn.pos == (i, j):
                    print("👑", end="")
                elif ally_pawn.pos == (i, j):
                    print("🤴", end="")
                elif enemy_1_pawn.pos == (i, j):
                    print("👿", end="")
                elif enemy_2_pawn.pos == (i, j):
                    print("👿", end="")
                else:
                    level = self.board.board[i][j]
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
        # print(
        #     "Player level:", self.board.board[playing_pawn.pos[0]][playing_pawn.pos[1]]
        # )

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

    user_input = True  # True if the user plays, False if random plays
    nb_games_todo = 1000
    nb_games = 0
    render = True
    delay = 0

    def get_random_action(possible_actions):
        sleep(delay)
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
    env = Env(test=False)
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
        if render:
            env.render()

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
            if render:
                env.render()
            env.stats()
            env.reset()
            nb_games += 1

            if nb_games == nb_games_todo:
                break
            # break
