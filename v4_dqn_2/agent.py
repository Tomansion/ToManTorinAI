import torch
import random
from collections import deque
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100000
BATCH_SIZE = 1000
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 0.996
LR = 0.001


class Agent:
    def __init__(self, nb_states, nb_actions):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.n_games = 0
        self.epsilon = EPS_START
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        
        self.model = Linear_QNet(nb_states, 256, nb_actions)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, train=False):
        # random moves: tradeoff exploration / exploitation
        final_move = [0] * self.nb_actions
        if train and random.random() < self.epsilon:
            move = random.randint(0, self.nb_actions - 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = int(torch.argmax(prediction).item())
            final_move[move] = 1

        return final_move

    def decrease_epsilon(self):
        if self.epsilon > EPS_END:
            self.epsilon *= EPS_DECAY

    def save(self):
        torch.save(self.model.state_dict(), "models/model.pth")

    def load(self):
        self.model.load_state_dict(torch.load("models/model.pth"))
