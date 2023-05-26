import torch
from env import Env
from agent import Agent


def test():
    from time import sleep

    env = Env()
    agent = Agent(env.get_state_size(), env.get_action_size())
    agent.load()
    state = env.reset()
    score_total = 0
    episodes = 1000

    for _ in range(episodes):
        while True:
            state = env.get_state()
            action = agent.get_action(state)
            reward, done, score = env.step(action)

            env.render()
            sleep(0.2)

            if done:
                break

        env.reset()
        score_total += score
        print(score_total)
        print("Average score:", score_total / episodes)

    print("Average score over", episodes, "episodes:", score_total / episodes)


if __name__ == "__main__":
    test()
