import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from random import randint
from keras.utils import to_categorical
import random
import statistics
from env_test_3 import Env, num_states, num_actions
from time import sleep

env = Env()


#################################
#   Define parameters manually  #
#################################
def define_parameters():
    params = dict()
    # Neural Network
    params["input_dim"] = num_states
    params["output_dim"] = num_actions
    params["epsilon_decay"] = 0.999
    params["learning_rate"] = 0.0005
    params["first_layer_size"] = 100  # neurons in the first layer
    params["second_layer_size"] = 300  # neurons in the second layer
    params["third_layer_size"] = 50  # neurons in the third layer
    params["episodes"] = 2000
    params["memory_size"] = 300
    params["batch_size"] = 30
    # Settings
    params["weights_path"] = "weights/weights7.hdf5"
    params["load_weights"] = False
    params["train"] = True
    params["plot_score"] = True
    params["delay_between_moves"] = 0
    return params


# Env_test, weights2: 10 :>
# Env_test_2, weights3: -8.5
# Env_test_2, weights4: -9.74, -5.8, -4.9, -3.85


def get_mean_stdev(array):
    return statistics.mean(array), statistics.stdev(array)


def test(params):
    params["load_weights"] = True
    params["train"] = False
    # Env_test_2, weights3: -8.5
    params["episodes"] = 100
    print("Testing...")
    score = run(params)
    return score


def run(params):
    agent = DQNAgent(params)
    weights_filepath = params["weights_path"]

    if params["load_weights"]:
        agent.model.load_weights(weights_filepath)
        print("weights loaded")

    nb_games = 0
    rewards_list = []
    turn_since_last_train = 0
    while nb_games < params["episodes"]:
        # Initialize
        print(f"==== Game {nb_games + 1}/{params['episodes']}")
        env.reset()
        done = False
        cumulated_reward = 0
        nb_moves = 0
        previous_rewards = 0

        # Decay epsilon
        if params["train"]:
            # agent.epsilon is set to give randomness to actions
            agent.epsilon *= params["epsilon_decay"]
            print(" - Epsilon:", agent.epsilon)
        else:
            agent.epsilon = 0.00

        while done is False:
            # get old state
            state_old = np.asarray(env.get_state())
            # print(" - State:", state_old)

            # perform random actions based on agent.epsilon, or choose the action
            if random.uniform(0, 1) < agent.epsilon:
                final_move = to_categorical(
                    randint(0, num_actions - 1), num_classes=num_actions
                )
            else:
                # predict action based on the old state
                prediction = agent.model.predict(state_old.reshape((1, num_states)))
                # print(" - prediction:", prediction)
                final_move = to_categorical(
                    np.argmax(prediction[0]), num_classes=num_actions
                )

            print(" - final_move:", final_move)

            # perform new move and get new state
            action_index = np.argmax(final_move)
            state_new, reward, done = env.step(action_index)
            state_new = np.asarray(state_new)
            print(" - Reward:", previous_rewards, " -> ", reward)

            if params["train"]:
                # train short memory base on the new action and state
                # print(" - Training short memory")
                agent.train_short_memory(state_old, final_move, reward, state_new, done)
                # store the new data into a long term memory
                # Only if last and current rewards are different than 0
                if previous_rewards != 0 or reward != 0:
                    agent.remember(state_old, final_move, reward, state_new, done)
                else:
                    if random.random() < 0.05:
                        agent.remember(state_old, final_move, reward, state_new, done)
            else:
                env.display()
                sleep(params["delay_between_moves"])

            cumulated_reward += reward
            nb_moves += 1
            previous_rewards = reward
            turn_since_last_train += 1

        # Game over
        env.display()
        rewards_list.append(cumulated_reward)
        print(f"== Game {nb_games + 1}/{params['episodes']}")
        if nb_games != 0:
            average_reward, stdev = get_mean_stdev(rewards_list[-20:])
            print("Average reward:", average_reward)
        nb_games += 1

        # Train long memory
        if params["train"] and turn_since_last_train > 10:
            # train long memory
            print("training long memory")
            agent.replay_new(agent.memory, params["batch_size"])

            # Save weights
            agent.model.save_weights(params["weights_path"], overwrite=True)
            turn_since_last_train = 0

    if params["train"]:
        # Test the trained model
        average_reward = test(params)
        print("Test average reward:", average_reward)
    else:
        average_reward, stdev = get_mean_stdev(rewards_list)
        return average_reward


if __name__ == "__main__":
    # Set options to activate or deactivate the game view, and its speed
    parser = argparse.ArgumentParser()
    params = define_parameters()
    args = parser.parse_args()
    params["bayesian_optimization"] = False  # Use bayesOpt.py for Bayesian Optimization
    run(params)
