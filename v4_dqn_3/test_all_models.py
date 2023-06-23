import os
from santorinai.tester import Tester
from santorinai.player_examples.random_player import RandomPlayer
from santorinai.player_examples.first_choice_player import FirstChoicePlayer
from santorinai.player_examples.basic_player import BasicPlayer
from santorinai_player import ToMantoRinAI


NB_GAMES = 100

MODEL_PATH = "models/"

model_list = os.listdir(MODEL_PATH)
ai_list = [RandomPlayer(), FirstChoicePlayer(), BasicPlayer()]

for model in model_list:
    ai_list.append(ToMantoRinAI(model_name=model))


# Init the tester
tester = Tester()
tester.verbose_level = 0  # 0: no output, 1: Each game results, 2: Each move summary
tester.delay_between_moves = 0  # Delay between each move in seconds
tester.display_board = False  # Display a graphical view of the board in a window

total_tests = len(ai_list) * (len(ai_list) - 1)
test_nb = 0

# Init a 2D array to store the results
results = [["-" for _ in range(len(ai_list))] for _ in range(len(ai_list))]
columns = []


# Profiling
import pstats
import cProfile

profiler = cProfile.Profile()
profiler.enable()


for i, p1 in enumerate(ai_list):
    p1_name = p1.name()
    columns.append(p1_name)

    for j, p2 in enumerate(ai_list):
        if i == j:
            continue

        # try:
        #     p2.reset_info()
        # except Exception as e:
        #     pass

        print(f"  {test_nb}/{total_tests}")

        # Play 100 games
        res = tester.play_1v1(p1, p2, nb_games=NB_GAMES)
        nb_win = res[p1_name]
        print(f"{p1.name()} vs {p2.name()}: {nb_win}/{NB_GAMES}")
        results[i][j] = int(100 * nb_win / NB_GAMES)

        test_nb += 1

        # try:
        #     p2.print_info()
        # except Exception as e:
        #     pass

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats(pstats.SortKey.TIME)
stats.dump_stats(filename="stats.prof")


# Save the results as a csv file
import csv

with open("tests/results.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow([""] + columns)
    for i in range(len(results)):
        writer.writerow([columns[i]] + results[i])

# Save the results as a markdown table
with open("tests/results.md", "w") as f:
    f.write("| |")
    for col in columns:
        f.write(f" {col} |")
    f.write("\n")
    f.write("|")
    for _ in range(len(columns) + 1):
        f.write(" --- |")
    f.write("\n")
    for i in range(len(results)):
        f.write(f"| {columns[i]} |")
        for j in range(len(results[i])):
            f.write(f" {results[i][j]} |")
        f.write("\n")

# Plot the results
import plotly.graph_objects as go

fig = go.Figure(data=go.Heatmap(z=results, x=columns, y=columns))
# Save the figure
fig.write_image("tests/results.png")

# Add the number in the cells
fig.update_layout(
    annotations=[
        go.layout.Annotation(
            x=j,
            y=i,
            text=str(results[i][j]),
            showarrow=False,
            font=dict(color="black"),
        )
        for i in range(len(results))
        for j in range(len(results[i]))
    ]
)

# Save the figure
fig.write_image("tests/results_with_numbers.png")
