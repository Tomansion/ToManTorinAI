from env import Env
from helper import plot, plot_test
from agent import Agent
from test import test


def train():
    current_episode = 0
    best_score_avg = 0
    plot_x = []
    plot_scores = []
    plot_mean_scores = []
    plot_mean_test_scores_test = []
    # plot_mean_test_scores_train = []
    episodes = 50000
    env = Env(test=True)

    agent = Agent(env.get_state_size(), env.get_action_size(), "agent")
    agent.save()

    print("NB_STATES:", env.get_state_size())
    print("NB_ACTIONS:", env.get_action_size())

    while True:
        # get old state
        state_old, possible_actions = env.get_state()

        # perform Action
        actions_choice = agent.get_action(state_old, possible_actions, train=True)
        reward, game_done = env.move(actions_choice)

        # train short memory and remember
        state_new, _ = env.get_state()
        agent.train_short_memory(
            state_old, actions_choice, reward, state_new, game_done
        )
        agent.remember(state_old, actions_choice, reward, state_new, game_done)

        if game_done:
            # train long memory, plot result
            agent.n_games += 1
            agent.train_long_memory()
            agent.decrease_epsilon()

            if reward > 0:
                plot_scores.append(1)
            else:
                plot_scores.append(0)
            mean_score = sum(plot_scores[-1000:]) / min(len(plot_scores), 1000)
            plot_mean_scores.append(mean_score)

            current_episode += 1

            # Save model and test
            if current_episode % 50 == 0:
                print(
                    "Game",
                    agent.n_games,
                    "Average Score:",
                    round(mean_score, 2),
                    "Average Best :",
                    best_score_avg,
                    "Epsilon:",
                    agent.epsilon,
                )
                agent.save()
                average_test_score_test = test(episodes=500, test=True)
                # average_test_score_train = test(episodes=250, test=False)
                plot_x.append(current_episode)
                plot_mean_test_scores_test.append(average_test_score_test)
                # plot_mean_test_scores_train.append(average_test_score_train)
                plot(plot_mean_scores)
                plot_test(plot_x, plot_mean_test_scores_test)

                if average_test_score_test >= best_score_avg:
                    best_score_avg = average_test_score_test
                    agent.save("best_move_agent")

            if current_episode >= episodes:
                return True

            env.reset()


if __name__ == "__main__":
    # Profiling
    import pstats
    import cProfile

    # profiler = cProfile.Profile()
    # profiler.enable()

    train()

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.dump_stats(filename="training_stats.prof")
