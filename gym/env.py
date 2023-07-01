import gymnasium as gym
import json
import numpy as np
from random import randint, choice, shuffle
from gymnasium import spaces

from santorinai.board import Board
from santorinai.player_examples import random_player, basic_player, first_choice_player
from santorinai.board_displayer.board_displayer import init_window, update_board
from helper import action_from_pos, new_pos_from_action

with open("config.json", "r") as f:
    conf = json.load(f)

BOARD_SIZE = 5
possible_enemy = []

if conf["enemy"]["random"]:
    possible_enemy.append(random_player.RandomPlayer())
if conf["enemy"]["first_choice"]:
    possible_enemy.append(first_choice_player.FirstChoicePlayer())
if conf["enemy"]["basic"]:
    possible_enemy.append(basic_player.BasicPlayer())


class Santorini(gym.Env):
    def __init__(self, render=False):
        self.action_space = spaces.Discrete(8 * 8)

        # Observation space consists of three 5x5 planes, represented as Box(3, 5, 5).
        # The first 5x5 plane is 1 for the agent's worker pieces and 0 otherwise.
        # The second 5x5 plane is 1 for the opponent's worker pieces and 0 otherwise.
        # The third 5x5 plane represents the height of the board at a given cell
        # in the grid - this ranges from 0 (no buildings) to 4 (dome).
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(3, 5, 5), dtype=np.int8
        )

        # Stats
        self.nb_win = 0
        self.nb_loose = 0
        self.nb_stuck_self = 0
        self.nb_stuck_other = 0

        # Rendering
        self.do_render = render
        if render:
            self.window = init_window(["AI", "Enemy"])

    def _get_obs(self):
        obs = np.zeros((3, 5, 5), dtype=np.int8)
        board = self.board.board

        playing_pawn = self.board.get_playing_pawn()
        playing_pawn_nb = playing_pawn.number
        ally_pawn_nb = (playing_pawn_nb + 2) % 4
        ally_pawn = self.board.pawns[ally_pawn_nb - 1]
        enemy_1_pawn_nb = (playing_pawn_nb + 1) % 4
        enemy_2_pawn_nb = (playing_pawn_nb + 3) % 4
        enemy_1_pawn = self.board.pawns[enemy_1_pawn_nb - 1]
        enemy_2_pawn = self.board.pawns[enemy_2_pawn_nb - 1]

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                # First plane
                if (i, j) == playing_pawn.pos:
                    obs[0, i, j] = 1
                elif (i, j) == ally_pawn.pos:
                    obs[0, i, j] = 2
                # Second plane
                if (i, j) == enemy_1_pawn.pos:
                    obs[1, i, j] = 1
                elif (i, j) == enemy_2_pawn.pos:
                    obs[1, i, j] = 2

                # Third plane
                obs[2, i, j] = board[i][j]

        return obs

    def _get_info(self):
        return {
            "nb_win": self.nb_win,
            "nb_loose": self.nb_loose,
            "nb_stuck_self": self.nb_stuck_self,
            "nb_stuck_other": self.nb_stuck_other,
        }

    def reset(self, seed=None, options=None):
        self.board = Board(2)

        # Select random enemy
        self.enemy = choice(possible_enemy)

        # Place pawns
        possible_positions = list(range(BOARD_SIZE**2))
        shuffle(possible_positions)

        pawn_pos_1 = possible_positions.pop()
        pawn_pos_2 = possible_positions.pop()
        enemy_pos_1 = possible_positions.pop()
        enemy_pos_2 = possible_positions.pop()

        self.board.pawns[0].pos = (pawn_pos_1 // BOARD_SIZE, pawn_pos_1 % BOARD_SIZE)
        self.board.pawns[2].pos = (pawn_pos_2 // BOARD_SIZE, pawn_pos_2 % BOARD_SIZE)
        self.board.pawns[1].pos = (enemy_pos_1 // BOARD_SIZE, enemy_pos_1 % BOARD_SIZE)
        self.board.pawns[3].pos = (enemy_pos_2 // BOARD_SIZE, enemy_pos_2 % BOARD_SIZE)

        # Don't always start first
        if randint(0, 1) == 0:
            self._move_enemy()

        return self._get_obs(), self._get_info()

    def step(self, action):
        move_action = int(action // 8)
        build_action = int(action % 8)
        self.playing_pawn = self.board.get_playing_pawn()
        new_move_pos = new_pos_from_action(self.playing_pawn.pos, move_action)
        new_build_pos = new_pos_from_action(new_move_pos, build_action)

        done = False
        reward = -1

        # ====== Move ======
        valid_move, reason = self.board.play_move(new_move_pos, new_build_pos)

        if self.do_render:
            self.render()

        if not valid_move:
            print("Invalid move, reason:", reason)
            reward = -5
            done = True

        # check goal
        elif self.board.is_game_over():
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

        return self._get_obs(), reward, done, False, self._get_info()

    def action_masks(self):
        playing_pawn = self.board.get_playing_pawn()
        possible_moves_pos = self.board.get_possible_movement_and_building_positions(
            playing_pawn
        )
        possible_actions = [0 for _ in range(8 * 8)]

        for move_pos, build_pos in possible_moves_pos:
            move_action = action_from_pos(playing_pawn.pos, move_pos)
            build_action = action_from_pos(move_pos, build_pos)
            combined_action = move_action * 8 + build_action
            possible_actions[combined_action] = 1

        return possible_actions

    def _move_enemy(self):
        if self.enemy:
            try:
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

            except Exception as e:
                print("Error in enemy move: ", e)

    def _is_stuck(self):
        playing_pawn = self.board.get_playing_pawn()
        possible_actions = self.board.get_possible_movement_and_building_positions(
            playing_pawn
        )
        return len(possible_actions) == 0

    def render(self):
        if self.do_render:
            from time import sleep

            update_board(self.window, self.board)
            sleep(1)
