import tensorflow as tf
import numpy as np
from utils import board_to_state, action_number_to_pos

from santorinai.board import Board
from santorinai.player_examples.random_player import RandomPlayer
from santorinai.player_examples.first_choice_player import FirstChoicePlayer
from santorinai.board_displayer.board_displayer import init_window, update_board


# Define the actor and critic networks using TensorFlow
class ActorNetwork(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation="relu")
        self.dense2 = tf.keras.layers.Dense(32, activation="relu")
        self.output_layer = tf.keras.layers.Dense(output_dim, activation="softmax")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)


class CriticNetwork(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(CriticNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(output_dim, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)


# Define the DDPG agent
class DDPGAgent:
    def __init__(
        self, state_dim, action_dim, actor_lr=0.001, critic_lr=0.001, gamma=0.99
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma

        self.actor_network = ActorNetwork(self.state_dim, self.action_dim)
        self.critic_network = CriticNetwork(self.state_dim, self.action_dim)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = self.actor_network(state)
        action = tf.random.categorical(tf.math.log(action_probs), 1)[0]
        return action.numpy()[0]

    def update(self, state, action, reward, next_state, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = tf.convert_to_tensor([action], dtype=tf.float32)
        reward = tf.convert_to_tensor([reward], dtype=tf.float32)
        next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.actor_network(next_state)
            target_q = reward + self.gamma * self.critic_network(next_state)
            critic_value = self.critic_network(state)
            critic_loss = tf.math.reduce_mean(tf.square(target_q - critic_value))

        critic_grad = tape.gradient(
            critic_loss, self.critic_network.trainable_variables
        )
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_network.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_network(state)
            critic_value = self.critic_network(state)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_network.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_network.trainable_variables)
        )


class Env:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = Board(2)
        self.random_player = RandomPlayer()
        self.player_number = 1 if np.random.rand() < 0.5 else 2
        self.opponent_player_number = 2 if self.player_number == 1 else 1
        print("New game started!")
        print("Our player number:", self.player_number)

        # Place the pawns randomly
        for i in range(4):
            playing_pawn = self.board.get_playing_pawn()
            positions = self.board.get_possible_movement_positions(playing_pawn)
            self.board.place_pawn(positions[np.random.randint(len(positions))])

    def get_state(self):
        return board_to_state(self.board)

    def step(self, action):
        # Play our player
        print("Our action:", action)
        move_pos = action_number_to_pos(self.board, action)
        board_copy = self.board.copy()
        playing_pawn_copy = board_copy.get_playing_pawn()
        playing_pawn_copy.move(move_pos)
        # Get random build position
        build_pos = board_copy.get_possible_building_positions(playing_pawn_copy)
        print("Possible build positions:", build_pos)
        random_build_pos = build_pos[np.random.randint(len(build_pos))]
        move_ok, error = self.board.play_move(move_pos, random_build_pos)

        if not move_ok:
            print("Wrong move:", error)
            self.board.get_playing_pawn().move(move_pos)
            self.board.board[random_build_pos[0]][random_build_pos[1]] += 1
            return self.get_state(), -5, True

        if self.board.winner_player_number == self.player_number:
            print("We won!")
            return self.get_state(), 1, True
        elif self.board.is_game_over():
            print("Draw!")
            return self.get_state(), 0, True

        # Play random player
        playing_pawn = self.board.copy().get_playing_pawn()
        move, build = self.random_player.play_move(self.board, playing_pawn)
        self.board.play_move(move, build)

        if self.board.winner_player_number == self.opponent_player_number:
            print("We lost!")
            return self.get_state(), -1, True
        elif self.board.is_game_over():
            print("Draw!")
            return self.get_state(), 0, True

        return self.get_state(), 0, False


# Initialize DDPG agent
state_dim = 225
action_dim = 8
env = Env()

agent = DDPGAgent(state_dim, action_dim)

# Training loop
total_episodes = 10000
batch_size = 64


for episode in range(total_episodes):
    env.reset()
    state = env.get_state()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Test the trained agent

state = env.reset()
done = False

print("Testing the trained agent...")
while not done:
    action = agent.get_action(state)
    next_state, reward, done = env.step(action)
    state = next_state

print(reward)
print("Done!")
# The agent has now learned to play Tic Tac Toe using the DDPG algorithm
