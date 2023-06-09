from env import Env
from helper import plot, plot_test
from agent import Agent
from test import test

current_episode = 0
best_score_avg = 0


def train():
    global current_episode, best_score_avg
    plot_scores = []
    plot_mean_scores = []
    plot_mean_test_scores_test = []
    # plot_mean_test_scores_train = []
    episodes = 50000
    env = Env(test=True)

    move_agent = Agent(env.get_state_size(), env.get_action_size(), "move_agent")
    build_agent = Agent(env.get_state_size(), env.get_action_size(), "build_agent")
    move_agent.save()
    build_agent.save()

    print("NB_STATES:", env.get_state_size())
    print("NB_ACTIONS:", env.get_action_size())

    while True:
        move_done, build_done = False, False
        # get old state
        move_state_old, possible_moves = env.get_move_state()

        # === Move
        # perform move
        move_choice = move_agent.get_action(move_state_old, possible_moves, train=True)
        move_reward, move_done = env.move(move_choice)
        # train short memory and remember
        move_state_new, _ = env.get_move_state()
        move_agent.train_short_memory(
            move_state_old, move_choice, move_reward, move_state_new, move_done
        )
        move_agent.remember(
            move_state_old, move_choice, move_reward, move_state_new, move_done
        )

        # === Build
        if not move_done:
            build_state_after_move, possible_moves = env.get_build_state()
            # perform build
            build_choice = build_agent.get_action(
                build_state_after_move, possible_moves, train=True
            )
            build_reward, build_done = env.build(build_choice)
            # train short memory and remember
            build_state_new, _ = env.get_build_state()
            build_agent.train_short_memory(
                build_state_after_move,
                build_choice,
                build_reward,
                build_state_new,
                build_done,
            )
            build_agent.remember(
                build_state_after_move,
                build_choice,
                build_reward,
                build_state_new,
                build_done,
            )

        if move_done and move_reward > 0:
            # Win
            build_agent.change_last_memory_reward(move_reward)

        def done():
            global current_episode, best_score_avg
            # train long memory, plot result
            move_agent.n_games += 1
            move_agent.train_long_memory()
            move_agent.decrease_epsilon()

            build_agent.n_games += 1
            build_agent.train_long_memory()
            build_agent.decrease_epsilon()

            score = env.score
            plot_scores.append(score)
            mean_score = sum(plot_scores[-1000:]) / 1000
            plot_mean_scores.append(mean_score)

            print(
                "Game",
                move_agent.n_games,
                "Score",
                score,
                "Average Score:",
                round(mean_score, 2),
                "Average Best :",
                best_score_avg,
                "Epsilon:",
                move_agent.epsilon,
            )

            current_episode += 1

            # Save model and test
            if current_episode % 50 == 0:
                move_agent.save()
                build_agent.save()
                average_test_score_test = test(episodes=500, test=True)
                # average_test_score_train = test(episodes=250, test=False)
                plot_mean_test_scores_test.append(average_test_score_test)
                # plot_mean_test_scores_train.append(average_test_score_train)
                plot(plot_scores, plot_mean_scores)
                plot_test(plot_mean_test_scores_test)

                if average_test_score_test >= best_score_avg:
                    best_score_avg = average_test_score_test
                    move_agent.save("best_move_agent")
                    build_agent.save("best_build_agent")

            if current_episode >= episodes:
                return True

            env.reset()

        if move_done or build_done:
            if done():
                break
        else:
            # Play enemy
            has_enemy_won = env.move_enemy()

            if has_enemy_won:
                # Set the models last reward to -10
                move_agent.change_last_memory_reward(-40)
                build_agent.change_last_memory_reward(-40)
                if done():
                    break


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