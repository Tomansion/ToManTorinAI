import tensorflow as tf
import numpy as np
from time import sleep
from train import env, get_actor, get_critic

lower_bound = -1000
upper_bound = 1000

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Loading the weights
actor_model.load_weights("pendulum_actor.h5")
critic_model.load_weights("pendulum_critic.h5")

target_actor.load_weights("pendulum_target_actor.h5")
target_critic.load_weights("pendulum_target_critic.h5")

total_episodes = 1000

for ep in range(total_episodes):
    prev_state = env.reset()
    episodic_reward = []

    while True:
        env.display()
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        sampled_actions = actor_model(tf_prev_state)
        # Get the highest value from the action.
        num_action = np.argmax(sampled_actions[0])

        # Recieve state and reward from environment.
        state, reward, done = env.step(num_action)

        sleep(0.1)
        

        episodic_reward.append(reward)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

    avg_reward = np.mean(episodic_reward)
    print("Total reward average: ", avg_reward)

