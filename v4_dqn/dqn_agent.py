from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add
import collections


class DQNAgent(object):
    def __init__(self, params):
        self.reward = 0
        self.gamma = 0.6
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params["learning_rate"]
        self.epsilon = 0.5
        self.actual = []
        self.input_dim = params["input_dim"]
        self.output_dim = params["output_dim"]
        self.first_layer = params["first_layer_size"]
        self.second_layer = params["second_layer_size"]
        self.third_layer = params["third_layer_size"]
        self.memory_size = params["memory_size"]
        self.memory = collections.deque(maxlen=params["memory_size"])
        self.weights = params["weights_path"]
        self.load_weights = params["load_weights"]
        self.model = self.network()

    def network(self):
        model = Sequential()
        model.add(Dense(self.first_layer, activation="relu", input_dim=self.input_dim))
        model.add(Dense(self.second_layer, activation="relu"))
        model.add(Dense(self.third_layer, activation="relu"))
        model.add(Dense(self.output_dim, activation="softmax"))
        opt = Adam(self.learning_rate)
        model.compile(loss="mse", optimizer=opt)

        if self.load_weights:
            model.load_weights(self.weights)
        return model

    def predict_action(self, state):
        self.q_values = self.model.predict(
            np.array(state).reshape(-1, *state.shape), verbose=0
        )[0]
        return self.q_values

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        print(f"Memory used: {len(self.memory)}/{self.memory_size}")

    def replay_new(self, memory, batch_size):
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory

        print(
            "Learning with minibatch of length:",
            len(minibatch),
            "out of",
            len(memory),
            "samples",
        )
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(np.array([next_state]), verbose=0)[0]
                )
            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(
                self.model.predict(next_state.reshape((1, self.input_dim)), verbose=0)[
                    0
                ]
            )
        target_f = self.model.predict(state.reshape((1, self.input_dim)), verbose=0)
        target_f[0][np.argmax(action)] = target
        self.model.fit(
            state.reshape((1, self.input_dim)), target_f, epochs=1, verbose=0
        )
