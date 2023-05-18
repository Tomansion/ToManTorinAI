from typing import Optional, Iterable

from v4_dqn.old.dqn_agent import DQNAgent
from statistics import mean
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("GTK3Agg")

# from keras.engine.saving import save_model

from env_test import Env, num_states, num_actions
from train import AgentConf

nb_games = 10000


# Run test
def test(conf: AgentConf):
    nb_win = 0
    nb_lose = 0
    turn_nb_win = []
    turn_nb_lose = []

    env = Env()

    agent = DQNAgent(
        num_states,
        n_neurons=conf.n_neurons,
        activations=conf.activations,
        epsilon=conf.epsilon,
        epsilon_min=conf.epsilon_min,
        epsilon_stop_episode=conf.epsilon_stop_episode,
        mem_size=conf.mem_size,
        discount=conf.discount,
        replay_start_size=conf.replay_start_size,
    )

    agent.load_model("model_dqn.hdf")

    for nb_game in range(nb_games):
        env.reset()
        done = False
        nb_turn = 1
        # game
        while not done:
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values(), train=False)
            print(nb_turn, best_state)
            # find the action, that corresponds to the best state
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break
            print("best_action:", best_action)

            _, reward, done = env.step(best_action)

            print("reward:", reward)

            if reward == 1:
                nb_win += 1
                turn_nb_win.append(nb_turn)
            elif reward == -1:
                nb_lose += 1
                turn_nb_lose.append(nb_turn)

            nb_turn += 1

    print("Win: {}, Lose: {}".format(nb_win, nb_lose))
    # Display ratio
    print("Win ratio: {}%".format(int((nb_win / nb_games) * 100)))
    print("Lose ratio: {}%".format(int((nb_lose / nb_games) * 100)))
    print(
        "Average win: {}, Average lose: {}".format(
            mean(turn_nb_win), mean(turn_nb_lose)
        )
    )


if __name__ == "__main__":
    conf = AgentConf()
    test(conf)


# Random :
# Win: 1658, Lose: 8342
# Win ratio: 16%
# Lose ratio: 83%
# Average win: 3.635102533172497, Average lose: 3.2628865979381443
