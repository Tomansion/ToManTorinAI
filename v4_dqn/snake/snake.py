import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from train2 import DQNAgent
from random import randint
from keras.utils import to_categorical
import random
import statistics


#################################
#   Define parameters manually  #
#################################
def define_parameters():
    params = dict()
    # Neural Network
    params["epsilon_decay_linear"] = 1 / 75
    params["learning_rate"] = 0.0005
    params["first_layer_size"] = 50  # neurons in the first layer
    params["second_layer_size"] = 300  # neurons in the second layer
    params["third_layer_size"] = 50  # neurons in the third layer
    params["episodes"] = 150
    params["memory_size"] = 2500
    params["batch_size"] = 1000
    # Settings
    params["weights_path"] = "weights/weights3.hdf5"
    params["load_weights"] = False
    params["train"] = True
    params["plot_score"] = True
    return params


class Game:
    def __init__(self, game_width, game_height):
        self.game_width = game_width
        self.game_height = game_height
        self.crash = False
        self.player = Player(self)
        self.food = Food()
        self.score = 0


class Player(object):
    def __init__(self, game):
        x = 0.45 * game.game_width
        y = 0.5 * game.game_height
        self.x = x - x % 20
        self.y = y - y % 20
        self.position = []
        self.position.append([self.x, self.y])
        self.food = 1
        self.eaten = False
        self.x_change = 20
        self.y_change = 0

    def update_position(self, x, y):
        if self.position[-1][0] != x or self.position[-1][1] != y:
            if self.food > 1:
                for i in range(0, self.food - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = x
            self.position[-1][1] = y

    def do_move(self, move, x, y, game, food, agent):
        move_array = [self.x_change, self.y_change]

        if self.eaten:
            self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1
        if np.array_equal(move, [1, 0, 0]):
            move_array = self.x_change, self.y_change
        elif (
            np.array_equal(move, [0, 1, 0]) and self.y_change == 0
        ):  # right - going horizontal
            move_array = [0, self.x_change]
        elif (
            np.array_equal(move, [0, 1, 0]) and self.x_change == 0
        ):  # right - going vertical
            move_array = [-self.y_change, 0]
        elif (
            np.array_equal(move, [0, 0, 1]) and self.y_change == 0
        ):  # left - going horizontal
            move_array = [0, -self.x_change]
        elif (
            np.array_equal(move, [0, 0, 1]) and self.x_change == 0
        ):  # left - going vertical
            move_array = [self.y_change, 0]
        self.x_change, self.y_change = move_array
        self.x = x + self.x_change
        self.y = y + self.y_change

        if (
            self.x < 20
            or self.x > game.game_width - 40
            or self.y < 20
            or self.y > game.game_height - 40
            or [self.x, self.y] in self.position
        ):
            game.crash = True
        eat(self, food, game)

        self.update_position(self.x, self.y)


class Food(object):
    def __init__(self):
        self.x_food = 240
        self.y_food = 200

    def food_coord(self, game, player):
        x_rand = randint(20, game.game_width - 40)
        self.x_food = x_rand - x_rand % 20
        y_rand = randint(20, game.game_height - 40)
        self.y_food = y_rand - y_rand % 20
        if [self.x_food, self.y_food] not in player.position:
            return self.x_food, self.y_food
        else:
            self.food_coord(game, player)


def eat(player, food, game):
    if player.x == food.x_food and player.y == food.y_food:
        food.food_coord(game, player)
        player.eaten = True
        game.score = game.score + 1


def get_record(score, record):
    if score >= record:
        return score
    else:
        return record


def initialize_game(player, game, food, agent, batch_size):
    state_init1 = agent.get_state(
        game, player, food
    )  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = [1, 0, 0]
    player.do_move(action, player.x, player.y, game, food, agent)
    state_init2 = agent.get_state(game, player, food)
    reward1 = agent.set_reward(player, game.crash)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_new(agent.memory, batch_size)


def get_mean_stdev(array):
    return statistics.mean(array), statistics.stdev(array)


def test(params):
    params["load_weights"] = True
    params["train"] = False
    score, mean, stdev = run(params)
    return score, mean, stdev


def run(params):
    agent = DQNAgent(params)
    weights_filepath = params["weights_path"]
    if params["load_weights"]:
        agent.model.load_weights(weights_filepath)
        print("weights loaded")
    counter_games = 0
    score_plot = []
    counter_plot = []
    record = 0
    total_score = 0
    while counter_games < params["episodes"]:
        # Initialize classes
        print(f"Game {counter_games + 1}/{params['episodes']}")
        game = Game(440, 440)
        player1 = game.player
        food1 = game.food

        # Perform first move
        initialize_game(player1, game, food1, agent, params["batch_size"])

        while not game.crash:
            if not params["train"]:
                agent.epsilon = 0.00
            else:
                # agent.epsilon is set to give randomness to actions
                agent.epsilon = 1 - (counter_games * params["epsilon_decay_linear"])

            # get old state
            state_old = agent.get_state(game, player1, food1)
            print("State:", state_old)

            # perform random actions based on agent.epsilon, or choose the action
            if random.uniform(0, 1) < agent.epsilon:
                final_move = to_categorical(randint(0, 2), num_classes=3)
            else:
                # predict action based on the old state
                prediction = agent.model.predict(state_old.reshape((1, 11)))
                print("prediction:", prediction)
                final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)
            print("final_move:", final_move)

            # perform new move and get new state
            player1.do_move(final_move, player1.x, player1.y, game, food1, agent)
            state_new = agent.get_state(game, player1, food1)

            # set reward for the new state
            reward = agent.set_reward(player1, game.crash)
            print("Reward", reward)

            if params["train"]:
                # train short memory base on the new action and state
                print("training short memory")
                agent.train_short_memory(
                    state_old, final_move, reward, state_new, game.crash
                )
                # store the new data into a long term memory
                agent.remember(state_old, final_move, reward, state_new, game.crash)

            record = get_record(game.score, record)

        print(f"Score: {game.score}")
        print(f"Max Score: {record}")

        if params["train"]:
            # train long memory
            print("training long memory")
            agent.replay_new(agent.memory, params["batch_size"])

        counter_games += 1
        total_score += game.score
        print(f"Game {counter_games}      Score: {game.score}")
        score_plot.append(game.score)
        counter_plot.append(counter_games)
    mean, stdev = get_mean_stdev(score_plot)

    if params["train"]:
        # Save weights when done
        agent.model.save_weights(params["weights_path"])
        total_score, mean, stdev = test(params)

    print("Total score: {}   Mean: {}   Std dev:   {}".format(total_score, mean, stdev))
    return total_score, mean, stdev


if __name__ == "__main__":
    # Set options to activate or deactivate the game view, and its speed
    parser = argparse.ArgumentParser()
    params = define_parameters()
    args = parser.parse_args()
    params["bayesian_optimization"] = False  # Use bayesOpt.py for Bayesian Optimization
    run(params)
