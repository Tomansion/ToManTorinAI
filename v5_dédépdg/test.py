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

episodic_reward = []
for ep in range(total_episodes):
    prev_state = env.reset()

    while True:
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        sampled_actions = actor_model(tf_prev_state)
        # Get the highest value from the action.
        num_action = np.argmax(sampled_actions[0])

        # Recieve state and reward from environment.
        state, reward, done = env.step(num_action)

        sleep(0.1)
        env.stats()
        env.display()
        

        episodic_reward.append(reward)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

    avg_reward = np.mean(episodic_reward)
    print("Total reward average: ", avg_reward)

# With 3000 episodes: 0.14
# With 4000 episodes: 0.17
# With 4000 and 8 actions: 0.20
# With 4000 and 8 actions + can't go warning: 0.31