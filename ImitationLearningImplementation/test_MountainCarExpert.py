import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym

actor = keras.models.load_model("PPOexpertMountainCar")

def sample_action(observation):
    logits = actor(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action




env = gym.make('MountainCar-v0')
episode_return=0
epochs=10

returns = []
for i in range(epochs):
    observation = env.reset()
    done = False
    episode_return=0
    while not done:
        env.render()
        observation = observation.reshape(1, -1)
        logits, action = sample_action(observation)
        observation, reward, done, _ = env.step(action[0].numpy())
        episode_return += reward
    returns.append(episode_return)


print(returns)

