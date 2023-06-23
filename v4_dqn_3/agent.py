import torch
import random
from collections import deque
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100000
BATCH_SIZE = 1000
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 0.9995
LR = 0.001


class Agent:
    def __init__(self, nb_states, nb_actions, name="agent"):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.name = name
        self.n_games = 0
        self.epsilon = EPS_START
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()

        self.model = Linear_QNet(nb_states, 256, nb_actions)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        self.total_choice = 0
        self.nb_failed = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )  # popleft if MAX_MEMORY is reached

    def change_last_memory_reward(self, reward):
        # change the last memory reward
        state, action, _, next_state, done = self.memory.pop()
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        if len(mini_sample) == 0:
            return

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, possible_moves, train=False):
        # random moves: tradeoff exploration / exploitation
        final_move = [0] * self.nb_actions
        if train and random.random() < self.epsilon:
            # Choose random move from possible moves
            # possible_moves : [0, 1, 0, 1, 0, 1, 0, ...]
            valid_moves = [i for i, x in enumerate(possible_moves) if x == 1]
            if len(valid_moves) == 0:
                final_move[0] = 1
                return final_move
            random_index = random.choice(valid_moves)
            final_move[random_index] = 1

        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            # choice = torch.argmax(prediction).item()
            # if possible_moves[choice] == 0:
            #     self.nb_failed += 1
            # self.total_choice += 1

            # Adding a constant to the prediction to avoid negative values
            prediction = prediction + abs(torch.min(prediction)) + 1
            # Masking the invalid moves
            prediction = prediction * torch.tensor(possible_moves, dtype=torch.float)
            move = int(torch.argmax(prediction).item())
            final_move[move] = 1

        return final_move

    def decrease_epsilon(self):
        if self.epsilon > EPS_END:
            self.epsilon *= EPS_DECAY

    def save(self, name=None):
        if name is None:
            torch.save(self.model.state_dict(), "models/" + self.name + ".pth")
        else:
            torch.save(self.model.state_dict(), "models/" + name + ".pth")

    def load(self):
        try:
            self.model.load_state_dict(torch.load("models/" + self.name + ".pth"))
            print("Model loaded")
        except FileNotFoundError:
            print("No model found, creating a new one")

    # def print_info(self):
    #     print("Failed moves: ", int(100 * self.nb_failed / self.total_choice))

    # def reset_info(self):
    #     self.total_choice = 0
    #     self.nb_failed = 0