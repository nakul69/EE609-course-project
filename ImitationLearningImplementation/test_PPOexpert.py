import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym

actor = keras.models.load_model("PPOexpert")

def sample_action(observation):
    logits = actor(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action




env = gym.make('CartPole-v1')
observation = env.reset()
done = False
episode_return=0

while not done:
    env.render()
    observation = observation.reshape(1, -1)
    logits, action = sample_action(observation)
    observation, reward, done, _ = env.step(action[0].numpy())
    episode_return += reward

print("Episode Return is : ",episode_return)

