from env import Env
from agent import Agent
from time import sleep


def test(delay=0.0, display=False, episodes=100, verbose=False):
    env = Env()
    agent = Agent(env.get_state_size(), env.get_action_size())
    agent.load()
    state = env.reset()
    score_total = 0
    print("Testing...")

    for i in range(episodes):
        while True:
            state = env.get_state()
            move = agent.get_action(state)
            reward, done, score = env.step(move)

            if display:
                env.render()
            sleep(delay)

            if done:
                break

        env.reset()
        score_total += score
        if verbose:
            print(score_total)
            print("Average score:", score_total / i)

    print("Average score over", episodes, "episodes:", score_total / episodes)
    env.stats()
    return score_total / episodes


if __name__ == "__main__":
    # test(delay=0.2, display=True, episodes=10)
    test(episodes=10000)