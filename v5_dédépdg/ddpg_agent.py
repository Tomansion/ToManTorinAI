import tensorflow as tf
from tensorflow import keras
import numpy as np

layers = keras.layers

# Define the environment
num_states = 10
num_actions = 5


# Define the actor model
def get_actor(num_states, num_actions):
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(num_actions, activation="softmax")(out)
    model = tf.keras.Model(inputs, outputs)
    return model


# Define the critic model
def get_critic(num_states, num_actions):
    state_input = layers.Input(shape=(num_states,))
    action_input = layers.Input(shape=(num_actions,))
    concat = layers.Concatenate()([state_input, action_input])
    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)
    model = tf.keras.Model([state_input, action_input], outputs)
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model


# Define the DDPG agent
class DDPGAgent:
    def __init__(self, num_states, num_actions):
        self.actor_model = get_actor(num_states, num_actions)
        self.critic_model = get_critic(num_states, num_actions)
        self.target_actor = get_actor(num_states, num_actions)
        self.target_critic = get_critic(num_states, num_actions)
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def get_action(self, state):
        state = np.reshape(state, [1, -1])
        action_probs = self.actor_model.predict(state)
        print("action_probs")
        print(action_probs)

        action = np.argmax(action_probs)
        return action

    def train(self, env, num_episodes=1000, batch_size=64, gamma=0.99, tau=0.005):
        replay_buffer = []
        for episode in range(num_episodes):
            print("Episode {}/{}".format(episode + 1, num_episodes))
            state = env.reset()
            done = False
            episode_reward = 0
            while not done:
                print(state)
                action = self.get_action(state)
                print(action)
                next_state, reward, done = env.step(action)
                episode_reward += reward
                replay_buffer.append((state, action, reward, next_state, done))
                state = next_state

                if len(replay_buffer) >= batch_size:
                    batch = np.random.choice(len(replay_buffer), size=batch_size, replace=False)
                    self._train_step(replay_buffer, batch, gamma, tau)
                    replay_buffer = []

            print("Episode {}: Reward = {}".format(episode, episode_reward))

    def _train_step(self, replay_buffer, batch, gamma, tau):
        states = np.array([replay_buffer[i][0] for i in batch])
        actions = np.array([replay_buffer[i][1] for i in batch])
        rewards = np.array([replay_buffer[i][2] for i in batch])
        next_states = np.array([replay_buffer[i][3] for i in batch])
        dones = np.array([replay_buffer[i][4] for i in batch])

        next_actions = self.target_actor.predict_on_batch(next_states)
        next_q_values = self.target_critic.predict_on_batch([next_states, next_actions]).flatten()
        targets = rewards + gamma * next_q_values * (1 - dones)

        self.critic_model.train_on_batch(x=[states, actions], y=targets)

        with tf.GradientTape() as tape:
            actions = self.actor_model(states)
            critic_value = self.critic_model([states, actions])
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

        self._update_target(tau)
