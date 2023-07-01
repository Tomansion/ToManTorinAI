import os
from santorinai.tester import Tester
from santorinai.player_examples.random_player import RandomPlayer
from santorinai.player_examples.first_choice_player import FirstChoicePlayer
from santorinai.player_examples.basic_player import BasicPlayer
from santorinai_player import ToMantoRinAI, ToMantoRinAIGuided


NB_GAMES = 1000

GUIDED = True
BASIC_ONLY = True
PROFILING = False

MODEL_PATH = "models/"

# Load all models
model_list = os.listdir(MODEL_PATH)
bot_list = [RandomPlayer(), FirstChoicePlayer(), BasicPlayer()]
ai_list = []
for model_name in model_list:
    if GUIDED:
        ai_list.append(ToMantoRinAIGuided(model_name))
    else:
        ai_list.append(ToMantoRinAI(model_name))

# Profiling
if PROFILING:
    import pstats
    import cProfile

    profiler = cProfile.Profile()
    profiler.enable()

# Init the tester
tester = Tester()
tester.verbose_level = 0  # 0: no output, 1: Each game results, 2: Each move summary
tester.delay_between_moves = 0  # Delay between each move in seconds
tester.display_board = False  # Display a graphical view of the board in a window

# ==== Play each AI against the basic AI ====

results = {}
for i, ai in enumerate(ai_list):
    ai_name = ai.name()
    print(f"  {i + 1}/{len(ai_list)} Testing {ai_name} against the basic AI")
    res = tester.play_1v1(ai, BasicPlayer(), nb_games=int(NB_GAMES / 2))
    nb_win = res[ai_name]
    res = tester.play_1v1(BasicPlayer(), ai, nb_games=int(NB_GAMES / 2))
    nb_win += res[ai_name]

    print(f"{ai_name} vs Basic AI: {nb_win}/{NB_GAMES}")
    results[ai_name] = int(100 * nb_win / NB_GAMES)

# Sort the results
basic_sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

# Print the results
print("\n\n")
for i, res in enumerate(basic_sorted_results):
    print(f"{i+1}. {res[0]}: {res[1]}%")


if BASIC_ONLY:
    exit(0)


# ==== Play each player against each other ====

# Add the bot to the player list
ai_list = bot_list + ai_list

total_tests = len(ai_list) * (len(ai_list) - 1)
test_nb = 0

# Init a 2D array to store the results
results = [["-" for _ in range(len(ai_list))] for _ in range(len(ai_list))]
columns = []


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
        # print(f"{p1.name()} vs {p2.name()}: {nb_win}/{NB_GAMES}")
        results[i][j] = int(100 * nb_win / NB_GAMES)

        test_nb += 1

        # try:
        #     p2.print_info()
        # except Exception as e:
        #     pass

if PROFILING:
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename="stats.prof")


# Calculate the average score for each player
avg_scores = {}
for i in range(len(ai_list)):
    avg_score = 0
    for j in range(len(ai_list)):
        if i < j:
            # We are P1
            avg_score += results[i][j]
        elif i > j:
            # We are P2
            avg_score += 100 - results[j][i]

    avg_scores[ai_list[i].name()] = avg_score / (len(ai_list) - 1)

# Sort the avg_scores
avg_score_results = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

# Print the average score
print("\n\nAverage score:")
for name, score in avg_score_results:
    print(f"{name}: {score:.2f}")

## === Export ===
export_name = "tests/results"
if GUIDED:
    export_name += "_guided"

# Save the results as a csv file
import csv

with open(export_name + ".csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow([""] + columns)
    for i in range(len(results)):
        writer.writerow([columns[i]] + results[i])

# Save the results as a markdown table
with open(export_name + ".md", "w") as f:
    f.write("# AI results comparison\n\n")
    if GUIDED:
        f.write("## Guided\n\n")
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

    # Add the average score
    f.write("\n\n# Average score\n\n")
    f.write("| | Score |\n")
    f.write("| --- | --- |\n")
    for name, score in avg_score_results:
        f.write(f"| {name} | {score:.2f} |\n")

    # Add the basic AI results
    f.write("\n\n# Basic AI results\n\n")
    f.write("| | Score |\n")
    f.write("| --- | --- |\n")
    for name, score in basic_sorted_results:
        f.write(f"| {name} | {score:.2f} |\n")

# Plot the results
import plotly.graph_objects as go

fig = go.Figure(data=go.Heatmap(z=results, x=columns, y=columns))

# Add the number in the cells
annotations = []

for i in range(len(results)):
    for j in range(len(results[i])):
        if i == j:
            continue

        annotations.append(
            dict(
                x=j,
                y=i,
                text=str(results[i][j]),
                showarrow=False,
                font=dict(color="black" if (results[i][j] > 50) else "white"),
            )
        )

fig.update_layout(annotations=annotations)

# Save the figure
fig.write_image(export_name + ".png", scale=2, width=1200, height=800)
