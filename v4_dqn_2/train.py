from env import Env
from helper import plot, plot_test
from agent import Agent
from test import test

def train():
    plot_scores = []
    plot_mean_scores = []
    plot_mean_test_scores = []
    total_score = 0
    episodes = 10000
    current_episode = 0
    env = Env()
    agent = Agent(env.get_state_size(), env.get_action_size())
    print("NB_STATES:", env.get_state_size())
    print("NB_ACTIONS:", env.get_action_size())

    while True:
        # get old state
        state_old = env.get_state()

        # get move
        final_move = agent.get_action(state_old, train=True)

        # perform move and get new state
        reward, done, score = env.step(final_move)
        state_new = env.get_state()

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            env.reset()
            agent.n_games += 1
            agent.train_long_memory()


            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            print(
                "Game",
                agent.n_games,
                "Score",
                score,
                "Average Score:",
                round(mean_score, 2),
                "Epsilon:",
                agent.epsilon,
            )
            # env.stats()

            agent.decrease_epsilon()

            current_episode += 1
            if current_episode >= episodes:
                break

            if current_episode % 10 == 0:
                agent.save()

            if current_episode % 50 == 0:
                average_test_score = test(episodes=300)
                plot_mean_test_scores.append(average_test_score)
                plot_test(plot_mean_test_scores)

if __name__ == "__main__":
    train()
