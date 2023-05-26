import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("GTK3Agg")


def plot(scores, mean_scores):
    plt.figure(2)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Number of games")
    plt.ylabel("Scores")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.pause(0.001)
