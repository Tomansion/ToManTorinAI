import gym
from gym import spaces
import numpy as np
from time import sleep

from random import randint, choice
import json
from env.helper import (
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

# ======= GYM ========

with open("config.json", "r") as f:
    conf = json.load(f)

BOARD_SIZE = 5

NB_ACTIONS = 8 * 8  # 8 directions + 8 build

FLASHLIGHT_SIZE = BOARD_SIZE - 1
FLASHLIGHT_BORDER_SIZE = FLASHLIGHT_SIZE * 2 - 1
FLASHLIGHT_AREA = FLASHLIGHT_BORDER_SIZE**2

NB_STATES = FLASHLIGHT_AREA * 5  # Tiles
NB_STATES += FLASHLIGHT_AREA  # Enemy pawn position
NB_STATES += 4  # Pawns level

print("NB_STATES:", NB_STATES)

possible_enemy = []

if conf["enemy"]["random"]:
    possible_enemy.append(random_player.RandomPlayer())
if conf["enemy"]["first_choice"]:
    possible_enemy.append(first_choice_player.FirstChoicePlayer())
if conf["enemy"]["basic"]:
    possible_enemy.append(basic_player.BasicPlayer())

for enemy in possible_enemy:
    print("Enemy:", enemy.name())


class Santorini(gym.Env):
    metadata = {"render_modes": ["santorinai", "emoticons"], "render_fps": 4}

    def __init__(self, test=False, render_mode=None):
        # Observation space is a list of 0 and 1, 0 if outside the board, 1 if inside
        # 0 if no pawn, 1 if pawn, ...

        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0, high=4, shape=(NB_STATES,), dtype=np.uint8
                )
            }
        )

        self.action_space = spaces.Discrete(NB_ACTIONS)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.test = test

        # Stats
        self.nb_win = 0
        self.nb_loose = 0
        self.nb_stuck_self = 0
        self.nb_stuck_other = 0

    def _get_obs(self):
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

        ## Tell what actions are possible
        # possible_actions = self._get_available_actions()

        return {"observation": np.array(state, dtype=np.uint8)}

    def _get_info(self):
        return {
            "nb_win": self.nb_win,
            "nb_loose": self.nb_loose,
            "nb_stuck_self": self.nb_stuck_self,
            "nb_stuck_other": self.nb_stuck_other,
        }

    def reset(self, seed=None, options=None):
        # Init board
        self.board = Board(2)
        # Select random enemy
        self.enemy = choice(possible_enemy)

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
            try:
                enemy_won = self._move_enemy()
                if enemy_won:
                    self.reset()
            except Exception as e:
                print(e)
                pass

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "emoticons":
            self.render()

        return observation, info

    def step(self, action):
        move_action = int(action // 8)
        build_action = int(action % 8)

        self.playing_pawn = self.board.get_playing_pawn()

        new_move_pos = new_pos_from_action(self.playing_pawn.pos, move_action)
        new_build_pos = new_pos_from_action(new_move_pos, build_action)

        # ====== Move ======
        self.board.play_move(new_move_pos, new_build_pos)

        done = False
        reward = -1

        # check goal
        if self.board.is_game_over():
            self.nb_win += 1
            reward = 10
            done = True

        # Check stuck
        elif self._is_stuck():
            self.nb_stuck_self += 1
            reward = -5
            done = True
        else:
            # ====== Enemy ======
            enemy_wins = self._move_enemy()
            if enemy_wins:
                self.nb_loose += 1
                reward = -10
                done = True

            # Check stuck
            elif self._is_stuck():
                self.nb_stuck_other += 1
                reward = -5
                done = True

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, False, info

    def render(self):
        # if not self.render_mode == "emoticons":
            # return    
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

        # Sleep
        sleep(0.5)

    def _is_stuck(self):
        playing_pawn = self.board.get_playing_pawn()
        possible_actions = self.board.get_possible_movement_and_building_positions(
            playing_pawn
        )
        return len(possible_actions) == 0

    def _move_enemy(self):
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

    def close(self):
        pass
