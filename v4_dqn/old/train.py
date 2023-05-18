from typing import Optional, Iterable

from v4_dqn.old.dqn_agent import DQNAgent
from datetime import datetime
from statistics import mean
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("GTK3Agg")

# from keras.engine.saving import save_model

from env_test import Env, num_states, num_actions


class AgentConf:
    def __init__(self):
        self.n_neurons = [32, 32]
        self.batch_size = 512
        self.activations = ["relu", "relu", "linear"]
        self.episodes = 1000
        self.epsilon = 1.0
        self.epsilon_min = 0.0
        self.epsilon_stop_episode = 500
        self.mem_size = 25000
        self.discount = 0.95
        self.replay_start_size = 2000
        self.epochs = 1
        self.render_every = None
        self.train_every = 1
        self.log_every = 5
        self.max_steps: Optional[int] = 10000


# Run dqn
def dqn(conf: AgentConf):
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

    scores = []
    scores_averages = []

    episodes_wrapped: Iterable[int] = tqdm(range(conf.episodes))
    for episode in episodes_wrapped:
        current_state = env.reset()
        done = False
        steps = 0

        # game
        while not done and (not conf.max_steps or steps < conf.max_steps):
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values())

            # find the action, that corresponds to the best state
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            _, reward, done = env.step(best_action)

            agent.add_to_memory(current_state, next_states[best_action], reward, done)
            current_state = next_states[best_action]
            steps += 1

        # just return score
        scores.append(env.get_game_score())

        # train
        if episode % conf.train_every == 0:
            # n = len(agent.memory)
            # print(f" agent.memory.len: {n}")
            agent.train(batch_size=conf.batch_size, epochs=conf.epochs)

        # logs
        if episode % conf.log_every == 0:
            avg_score = mean(scores[-conf.log_every :])
            min_score = min(scores[-conf.log_every :])
            max_score = max(scores[-conf.log_every :])
            episodes_wrapped.set_description("")
            print(f"\nepisode: {episode}/{conf.episodes}, ")
            print(f" - avg_score: {avg_score:.2f}, ")
            print(f" - min_score: {min_score:.2f}, ")
            print(f" - max_score: {max_score:.2f}, ")
            print(f" - epsilon: {agent.epsilon:.2f}")
            scores_averages.append(avg_score)

    # save_model
    # save_model(agent.model, f'{log_dir}/model.hdf', overwrite=True, include_optimizer=True)
    agent.save_model("model_dqn.hdf")

    plt.plot(scores_averages)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()


# def enumerate_dqn():
#     """ Enumerate hyper-params to find the best combination """
#     for mem_size in [10_000, 15_000, 20_000, 25_000]:
#         for epochs in [1, 2, 3]:
#             for epsilon_stop_episode in [1600, 1800, 2000]:
#                 for discount in [0.95, 0.97, 0.99]:
#                     conf = AgentConf()
#                     conf.mem_size = mem_size
#                     conf.epochs = epochs
#                     conf.epsilon_stop_episode = epsilon_stop_episode
#                     conf.discount = discount
#                     dqn(conf)


if __name__ == "__main__":
    conf = AgentConf()
    dqn(conf)
    # to avoid jump to console when run under IDE
    exit(0)
